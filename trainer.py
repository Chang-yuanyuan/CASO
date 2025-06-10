import os
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union, Tuple, List, Dict
import torch.optim as optim

from configs.config import get_args
from classifier.classifier import BinaryClassifier
# from dataset.ranker import *
from dataset.celeba_dataset import *
import random
from tqdm import tqdm
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from diffusers.optimization import get_scheduler

from utils.utils import *


class MLPModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=768, device=None, rank=None):
        super(MLPModel, self).__init__()
        # 定义两个 MLP 模型
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
        ).to(device)

        self.mlp2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
        ).to(device)

        # 如果使用 DDP
        if rank is not None:
            self.mlp1 = DDP(self.mlp1, device_ids=[rank])
            self.mlp2 = DDP(self.mlp2, device_ids=[rank])

    def forward(self, condition):
        embedding_1 = self.mlp1(condition)
        embedding_2 = self.mlp2(condition)
        return [embedding_1, embedding_2]

    def save_models(self, path1, path2):
        torch.save(self.mlp1.module if isinstance(self.mlp1, DDP) else self.mlp1, path1)
        torch.save(self.mlp2.module if isinstance(self.mlp2, DDP) else self.mlp2, path2)


def train(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # 加载模型，定义MLP load model, define mlp(to be trained)
    tokenizer, text_encoder, vae, unet, noise_scheduler = prepare_SD_modules(args.pretrained_model)

    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    unet = unet.to(device)

    classifier = torch.load(args.classifier_path, map_location=device)

    text_encoder_token_embedding = text_encoder.text_model.embeddings.token_embedding
    text_encoder_position_embedding = text_encoder.text_model.embeddings.position_embedding

    noise_scheduler.set_timesteps(args.NUM_DDIM_STEPS, device=device)
    timesteps = noise_scheduler.timesteps.to(device)

    mlp_model = MLPModel(device=device, rank=rank)

    # 冻结模型参数 Freezing all models except mlp
    freeze_params(text_encoder.parameters())
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    freeze_params(classifier.parameters())

    set_xformers(unet=unet)

    dataset = DiscoveryDataset(data_root_dir='/media/inspur/disk/yychang_workspace/data/VanGogh2photo/mix',
                               tokenizer=tokenizer,
                               size=args.size)
    # dataset = DiscoveryDataset(data_root_dir='/media/inspur/disk/yychang_workspace/data/vangogh2photo/mix',
    #                            tokenizer=tokenizer,
    #                            size=args.size)

    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)

    optimizer = optim.AdamW(mlp_model.parameters(),
                            lr=args.AdamW_lr * world_size,  # 学习率缩放
                            betas=(args.AdamW_betas_start, args.AdamW_betas_end),
                            weight_decay=args.AdamW_weight_decay,
                            eps=args.AdamW_eps)

    lr_scheduler = get_scheduler(name=args.lr_scheduler,
                                 optimizer=optimizer,
                                 num_warmup_steps=args.warmup_steps,
                                 num_training_steps=len(dataloader) * args.num_epochs)
    mse_loss = nn.MSELoss()
    if rank == 0:
        print(f"Start training...(temperature={args.temperature}, attribution=Smile)")
    batch_losses = []
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        total_num = 0
        mlp_model.train()
        with (tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", unit="batch", disable=rank != 0) as pbar):
            for batch in pbar:
                optimizer.zero_grad()
                pixel_values = batch['img_pixel_values'].to(device)
                latents = vae.encode(pixel_values).latent_dist.mean.detach()
                latents = latents * 0.18215
                z0 = latents

                uncond_text_ids = tokenizer(
                    ["" for _ in range(latents.shape[0])],  # 空文本，无条件预测
                    padding="max_length",  # 输入文本填充到模型所需的最大长度
                    truncation=True,  # 如果输入文本超过最大长度，则截断它
                    max_length=tokenizer.model_max_length,  # 使用分词器支持的最大长度
                    return_tensors="pt"  # 返回 PyTorch 张量格式的结果
                ).input_ids.to(latents.device)
                uncond_embedding = text_encoder(uncond_text_ids)[0]  # .repeat(latents.shape[0], 1, 1)

                nn_embedded = text_encoder_token_embedding(
                    uncond_text_ids.to(device))  # torch.Size([batch_size, 77, 768])

                # Set mlp embedding(to be trained)
                condition = torch.ones((latents.shape[0], 1), device=device)
                condition = condition.float()
                torch.distributed.barrier()
                cond_index = get_synced_cond_index(rank)

                cond_mlp_embedding = mlp_model(condition)[cond_index]

                # cond_mlp_embedding = mlp(condition)  # cond_mlp_embedding.shape:torch.Size([768])
                # cond_mlp_embedding = condition(args.batch_size)
                nn_embedded[:, 1, :] = cond_mlp_embedding
                position_ids = torch.arange(uncond_text_ids.size(1), dtype=torch.long, device=device).unsqueeze(0)

                position_embeds = text_encoder_position_embedding(position_ids)
                position_embeds = position_embeds.expand(nn_embedded.shape[0], -1, -1)
                cond_1 = position_embeds + nn_embedded

                assert nn_embedded.shape == position_embeds.shape
                causal_attention_mask = make_causal_mask(uncond_text_ids.shape, cond_1.dtype, device)

                cond_2 = text_encoder.text_model.encoder(cond_1,
                                                         causal_attention_mask=causal_attention_mask).last_hidden_state

                cond_embedding = text_encoder.text_model.final_layer_norm(cond_2)

                all_latent, all_times = ddim_forward_loop(latents,
                                                          noise_scheduler,
                                                          uncond_embedding,
                                                          args.NUM_DDIM_STEPS,
                                                          unet)

                # random_index = random.randint(10, 30)
                t = all_times[-30].detach()
                # selected_index = torch.where(timesteps == t)[0]
                # selected_timesteps = timesteps[selected_index:]
                noise = torch.randn(latents.shape).to(latents.device)

                noisy_latents = noise_scheduler.add_noise(z0, noise, t)

                noise_pred_cond = unet(noisy_latents, t, encoder_hidden_states=cond_embedding,
                                       cross_attention_kwargs={},
                                       return_dict=False
                                       )[0]
                noise_pred_uncond = unet(noisy_latents, t, encoder_hidden_states=uncond_embedding,
                                         cross_attention_kwargs={},
                                         return_dict=False
                                         )[0]

                noise_pred = noise_pred_uncond + args.GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                pred_z0 = pred_original(model_output=noise_pred, timestep=t, sample=noisy_latents,
                                        scheduler=noise_scheduler, device=device)

                latent_to_be_decode = pred_z0
                latent_to_be_decode = 1 / 0.18215 * latent_to_be_decode
                x0 = vae.decode(latent_to_be_decode).sample

                attr_label = classifier(x0)

                loss = 0 * mse_loss(pred_z0, z0) + args.temperature * ((cond_index - attr_label) ** 2).mean()
                loss.backward()

                if rank == 0:
                    batch_losses.append({
                        'epoch': epoch + 1,
                        'loss': loss.item()
                    })

                optimizer.step()
                lr_scheduler.step()
                running_loss += loss.item()
                total_num += 1
                pbar.set_postfix({"loss": running_loss / total_num})
                torch.distributed.barrier()

            if rank == 0:
                # torch.save(mlp.module, f'Horse_{args.temperature}_1.pt')
                mlp_model.save_models(f'vangogh_{args.temperature}_0_epoch{epoch}.pt',
                                      f'vangogh_{args.temperature}_1_epoch{epoch}.pt')
                # loss_df = pd.DataFrame(batch_losses)
                # excel_file = 'mustache_loss_num1.xlsx'
                # loss_df.to_excel(excel_file, index=False)

    cleanup()
