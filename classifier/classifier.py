import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,4,5'
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR
from dataset.celeba_dataset import *
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import DataLoader, Subset, DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import torch
import torch.distributed as dist


class BinaryClassifier(nn.Module):
    def __init__(self, model_type: str):
        super(BinaryClassifier, self).__init__()
        if model_type == "vgg16":
            self.model_type = "vgg16"
            self.model = models.vgg16(weights=None)
            self.model.load_state_dict(torch.load('/media/inspur/disk/yychang_workspace/pretrained_model/vgg16.pth'))
            self.model.classifier[6] = nn.Linear(4096, 1)  # 二分类设置为输出1个值
            for param in self.model.parameters():
                param.requires_grad = True
            for param in self.model.classifier[6].parameters():
                param.requires_grad = True
        elif model_type == "resnet50":
            self.model_type = "resnet50"
            self.model = models.resnet50(weights=None)
            self.model.load_state_dict(torch.load('/media/inspur/disk/yychang_workspace/pretrained_model/resnet50.pth'))
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, 1)
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True
        else:
            raise TypeError('请选择正确的模型：resnet50 or vgg16')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x


def prepare():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "")
    dataset = celeba_Dataset("Mustache")
    train_size = 15000

    remaining_size = len(dataset) - train_size

    val_size = int(0.5 * remaining_size)
    test_size = remaining_size - val_size

    # 使用 Subset 划分数据集
    train_dataset = Subset(dataset, list(range(train_size)))
    remaining_indices = list(range(train_size, len(dataset)))
    val_dataset = Subset(dataset, remaining_indices[:val_size])
    test_dataset = Subset(dataset, remaining_indices[val_size:])

    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    return device, train_loader, val_loader, test_loader


def train(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # 选择一个未被占用的端口

    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dataset = celeba_Dataset("Wavy_Hair", image_root_path='/media/inspur/disk/yychang_workspace/data/celeba_hq_latent')
    train_size = 15000
    remaining_size = len(dataset) - train_size
    val_size = int(0.5 * remaining_size)
    test_size = remaining_size - val_size

    train_dataset = Subset(dataset, list(range(train_size)))
    remaining_indices = list(range(train_size, len(dataset)))
    val_dataset = Subset(dataset, remaining_indices[:val_size])
    test_dataset = Subset(dataset, remaining_indices[val_size:])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=24, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, sampler=val_sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, sampler=test_sampler, num_workers=4, pin_memory=True)

    model = BinaryClassifier(model_type='vgg16')
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 对于 Sigmoid 激活函数，使用 BCEWithLogitsLoss
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.1)  # 每step_size个epoch将学习率降低为原来的gamma

    num_epochs = 8
    print("Start training...")
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        for batch in tqdm(train_loader, disable=rank != 0):
            optimizer.zero_grad()
            img = batch["img_pixel_values"].to(device, non_blocking=True)
            label = batch["img_label"].to(device, non_blocking=True)
            outputs = model(img)
            loss = criterion(outputs, label.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader.dataset)
        if rank == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # 验证模型
        if (epoch + 1) % 2 == 0:
            model.eval()
            corrects = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    img = batch["img_pixel_values"].to(device)
                    labels = batch["img_label"].to(device)
                    outputs = model(img)
                    preds = (outputs > 0.5).float()  # Sigmoid 的阈值为 0.5
                    total += labels.size(0)
                    corrects += torch.sum(preds == labels.data.unsqueeze(1))
            epoch_acc = corrects.double() / total
            if rank == 0:
                print(f'Validation Accuracy: {epoch_acc:.4f}')

        scheduler.step()
        if rank == 0:
            torch.save(model.module, f'classifier_Wavy_hair_latent300.pt')


def test():
    _, train_loader, val_loader, test_loader = prepare()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "")
    # model = BinaryClassifier("vgg16")
    model = torch.load(f'/media/inspur/disk/yychang_workspace/code/NoiseCLR/ranker/classifier_celebahq_Mustache.pt',
                       map_location=device)
    model.eval()
    total = 0
    total_0 = 0
    total_1 = 0
    right = 0
    right_0 = 0
    right_1 = 0
    with torch.no_grad():
        for batch in test_loader:
            img = batch["img_pixel_values"].to(device)
            labels = batch["img_label"].to(device)
            outputs = model(img)  # shape = [batch_size, 1]
            outputs = outputs.squeeze(1)
            assert len(labels) == len(outputs), "Index out of range"

            preds = torch.where(outputs > 0.5, torch.tensor(1.0, device=outputs.device),
                                torch.tensor(0.0, device=outputs.device))
            preds = preds.long()
            mask_0 = labels == 0
            mask_1 = labels == 1

            img_paths = batch["img_path"]

            mask_1_indices = mask_1.nonzero(as_tuple=True)[0]  # 获取 mask_1 为 True 的索引
            for idx in mask_1_indices:
                print(img_paths[idx])
            # break
            # # print(labels)
            # total += len(labels)
            # total_0 += torch.sum(mask_0)
            # right_0 += (preds[mask_0] == labels[mask_0]).sum().item()
            #
            # total_1 += torch.sum(mask_1)
            # right_1 += (preds[mask_1] == labels[mask_1]).sum().item()
            # result = torch.sum(preds == labels)
            # right += result.item()

        # print(f'Test Accuracy: {right / total:.4f}')
        # print(right_0, total_0, right_0 / total_0)
        # print(right_1, total_1, right_1 / total_1)
        # print(right, total, right / total)


#
def test_single(device, img_path):
    model = torch.load("/media/inspur/disk/yychang_workspace/code/NoiseCLR/classifier_celebahq_BlackHair.pt",
                       map_location=device)
    model.sigmoid = nn.Identity()

    model.eval()
    # img_path = '/media/inspur/disk/yychang_workspace/data/celebA/img_align_celeba/000050.jpg'

    image = Image.open(img_path)
    image = image.convert("RGB") if image.mode != "RGB" else image

    image_processed = image.resize((512, 512), resample=Image.Resampling.BICUBIC)

    image_processed = np.array(image_processed).astype(np.uint8)
    image_processed = (image_processed / 127.5 - 1.0).astype(np.float32)  # Pixel values between -1, 1
    image_processed = torch.from_numpy(image_processed).permute(2, 0, 1)  # HWC->CHW
    image_processed = image_processed.unsqueeze(0)
    image_processed = image_processed.to(device)
    preds = model(image_processed)
    print(preds)
    return preds.item()

if __name__ == '__main__':
    # world_size = torch.cuda.device_count()
    # mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    # dataset = celeba_Dataset("Black_Hair")
    # loader = DataLoader(dataset, batch_size=2560, shuffle=False, num_workers=0)
    # compute_proportion(loader)
    # # test()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "")
    # list_attr = []
    # image_paths = []
    # for root, dirs, files in os.walk("/media/inspur/disk/yychang_workspace/data/ffhq1024"):
    #     for file in sorted(files):
    #         file_extension = file.split(".")[-1]  # 分割文件名，取最后一个部分作为文件扩展名。
    #         if file_extension in VALID_IMG_FORMATS:
    #             image_paths.append(os.path.join(root, file))
    # correct = 0
    # for single_path in tqdm(image_paths):
    #     a = test_single(device, single_path)
    #     list_attr.append(a)
    # plt.hist(list_attr, edgecolor='black')  # bins 参数控制直方图的柱子数量
    # plt.show()
    #     image = Image.open(single_path)
    #     image = image.convert("RGB") if image.mode != "RGB" else image
    #     image_processed = image.resize((512,512), resample=Image.Resampling.BICUBIC)
    #     image_processed = np.array(image_processed).astype(np.uint8)
    #     image_processed = (image_processed / 127.5 - 1.0).astype(np.float32)  # Pixel values between -1, 1
    #     img = torch.from_numpy(image_processed).permute(2, 0, 1)  # HWC->CHW
    #     img = img.unsqueeze(0).to(device)
    #     model = torch.load(f'/media/inspur/disk/yychang_workspace/code/NoiseCLR/ranker/classifier_celebahq_Mustache.pt',
    #                        map_location=device)
    #
    #     # model = model.to(device)
    #
    #     model.eval()
    #     label = model(img)
    #     if label > 0.5:
    #         correct += 1
    # print(correct)
    # edit = test_mlp_single(single_path)
    # filename = os.path.basename(single_path)
    # save_path = os.path.join(save_root, filename)
    # edit.save(save_path)
    # print(f"save {save_path}")
    # device, train_loader, val_loader, test_loader = prepare()
    # train(device, train_loader, val_loader, test_loader)
    # test_single(device="cuda:5")

    # 处理不均衡数据
    dataset = celeba_Dataset(attribute="Pale_Skin")
    train_size = 15000
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    remaining_size = len(dataset) - train_size
    remaining_indices = list(range(train_size, len(dataset)))
    val_size = int(0.5 * remaining_size)

    train_dataset = Subset(dataset, list(range(train_size)))
    labels = [train_dataset.dataset.labels_dict[i][train_dataset.dataset.attribute_index] for i in
              range(len(train_dataset))]
    label_counts = Counter(labels)

    class_weights = {label: 1.0 / count for label, count in label_counts.items()}

    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=48, sampler=sampler)
    val_dataset = Subset(dataset, remaining_indices[:val_size])
    test_dataset = Subset(dataset, remaining_indices[val_size:])

    val_loader = DataLoader(val_dataset, batch_size=48, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False, num_workers=0)
    # train("cuda:5", train_loader, val_loader, test_loader)

    model = BinaryClassifier(model_type='vgg16')
    # model = torch.load('/media/inspur/disk/yychang_workspace/code/NoiseCLR/ranker/classifier_Bushy_Eyebrows.pt')
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 对于 Sigmoid 激活函数，使用 BCEWithLogitsLoss
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.1)  # 每step_size个epoch将学习率降低为原来的gamma

    num_epochs = 6
    print("Start training...")
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            img = batch["img_pixel_values"].to(device, non_blocking=True)
            label = batch["img_label"].to(device, non_blocking=True)
            outputs = model(img)
            loss = criterion(outputs, label.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader.dataset)

        if (epoch + 1) % 2 == 0:
            model.eval()
            corrects = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    img = batch["img_pixel_values"].to(device)
                    labels = batch["img_label"].to(device)
                    outputs = model(img)
                    preds = (outputs > 0.5).float()  # Sigmoid 的阈值为 0.5
                    total += labels.size(0)
                    corrects += torch.sum(preds == labels.data.unsqueeze(1))
            epoch_acc = corrects.double() / total

            print(f'Validation Accuracy: {epoch_acc:.4f}')

        scheduler.step()

        torch.save(model, f'classifier_Pale_Skin.pt')
