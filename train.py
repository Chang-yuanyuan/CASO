import os
import torch
import torch.multiprocessing as mp
from trainer import train

from configs.config import get_args
from classifier.classifier import BinaryClassifier

if __name__ == '__main__':
    args = get_args()
    args.temperature = 1
    # DDP train
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)

