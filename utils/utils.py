import torch
from typing import Optional, Union
import random
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
from diffusers.utils.import_utils import is_xformers_available
from torchvision import transforms
from diffusers.optimization import get_scheduler
import PIL
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from core.dataset import *
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def get_noise_pred_single(unet, latents, t, embeddings):
    noise_pred_single = unet(latents, t, encoder_hidden_states=embeddings)["sample"]
    return noise_pred_single


def prev_step(model_output: Union[torch.FloatTensor, np.ndarray],
              timestep: int,
              scheduler,
              sample: Union[torch.FloatTensor, np.ndarray]):
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[
        prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5  # 预测的x0
    pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
    prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
    return prev_sample, pred_original_sample


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, scheduler,
              sample: Union[torch.FloatTensor, np.ndarray]):
    timestep, next_timestep = min(
        timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = scheduler.alphas_cumprod[timestep] if timestep >= 0 else scheduler.final_alpha_cumprod
    alpha_prod_t_next = scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample  # ≈ add noise


def pred_original(model_output: Union[torch.FloatTensor, np.ndarray],
                  timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray],
                  scheduler,
                  device
                  ):
    alpha_prod_t = scheduler.alphas_cumprod.to(device)[timestep]
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5  # 预测的x0
    return pred_original_sample


@torch.no_grad()
def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents.detach()
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).astype(np.uint8)
    return image


@torch.no_grad()
def ddim_forward_loop(latent, scheduler, uncond_embeddings, NUM_DDIM_STEPS, unet):
    all_latent = [latent]
    all_times = [0]
    latent = latent.clone().detach()
    for i in range(NUM_DDIM_STEPS):
        t = scheduler.timesteps[len(scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(unet, latent, t, uncond_embeddings)
        latent = next_step(noise_pred, t, scheduler=scheduler, sample=latent)
        all_latent.append(latent)
        all_times.append(t)
    return all_latent, all_times


def set_xformers(unet):
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
    else:
        print("xformers is not available.")


def find_all_images(directory, extensions=(".jpg", ".png", ".jpeg")):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in sorted(files):
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


# 处理图像为模型可以输入的形式
def process_img(img_path=None, size=512, resample=Image.Resampling.BICUBIC, is_training=False):
    if not img_path:
        raise ValueError("img_path is required")
    image = Image.open(img_path)
    image = image.convert("RGB") if image.mode != "RGB" else image
    image_processed = image.resize((size, size), resample=resample)
    if is_training:
        image_processed = transforms.RandomHorizontalFlip(p=0.5)(image_processed)  # 以一定概率水平翻转图像
    image_processed = np.array(image_processed).astype(np.uint8)
    image_processed = (image_processed / 127.5 - 1.0).astype(np.float32)  # Pixel values between -1, 1
    img = torch.from_numpy(image_processed).permute(2, 0, 1)  # HWC->CHW
    img = img.unsqueeze(0)
    return img


def prepare_SD_modules(pretrained_model):
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet")
    noise_scheduler = PNDMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")
    return tokenizer, text_encoder, vae, unet, noise_scheduler


# DDP utils
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def get_synced_cond_index(rank):
    cond_index = torch.randint(0, 2, (1,), device=f"cuda:{rank}")  # 在 GPU 上生成随机数
    dist.broadcast(cond_index, src=0)  # 从主进程广播到所有进程
    return cond_index.item()


def get_synced_cond_index_muti(rank):
    cond_index = torch.randint(0, 3, (1,), device=f"cuda:{rank}")  # 在 GPU 上生成随机数
    dist.broadcast(cond_index, src=0)  # 从主进程广播到所有进程
    return cond_index.item()


def cleanup():
    dist.destroy_process_group()


def resize_img(org_path, save_path):
    img = Image.open(org_path)
    img = img.resize(512, 512, resample=Image.Resampling.BICUBIC)
    basename = os.path.basename(org_path)
    save_path = os.path.join(save_path, basename)
    img.save(save_path)

#
# def edit():
#     dataset = celeba_Dataset_single(get_label=0, attribute="Mustache")
#
#     dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)
#     args.GUIDANCE_SCALE = -10
