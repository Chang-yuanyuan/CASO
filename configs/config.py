import argparse
import os
from PIL import Image


def get_args():
    # root_path = "/slurm-files/cyy"
    root_path = "/media/inspur/disk/yychang_workspace"
    parser = argparse.ArgumentParser()
    # DDP args
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')

    # train args
    parser.add_argument('--NUM_DDIM_STEPS',
                        type=int, default=50,
                        help='Number of DDIM steps')
    parser.add_argument('--num_epochs',
                        type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--temperature',
                        type=float, default=1,
                        help='Temperature for sampling')
    parser.add_argument('--AdamW_lr',
                        type=float, default=1e-2,
                        help='Learning rate for AdamW optimizer')
    parser.add_argument('--AdamW_betas_start',
                        type=float, default=0.9,
                        help='Beta starting value for AdamW optimizer')
    parser.add_argument('--AdamW_betas_end',
                        type=float, default=0.999,
                        help='Beta ending value for AdamW optimizer')
    parser.add_argument('--AdamW_weight_decay',
                        type=float, default=1e-2,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--AdamW_eps',
                        type=float, default=1e-08,
                        help='Epsilon value for AdamW optimizer')
    parser.add_argument('--batch_size',
                        type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--size',
                        type=int, default=512,
                        help='Image size')
    parser.add_argument('--GUIDANCE_SCALE',
                        type=float, default=10,
                        help='Classifier free guidance scale')
    parser.add_argument('--interpolation',
                        default=Image.Resampling.BICUBIC,
                        help='Interpolation method')
    parser.add_argument('--attribution',
                        type=str, default='Smiling',
                        help='Attribution for classifier')
    parser.add_argument('--pretrained_model',
                        type=str,
                        default=os.path.join(root_path, 'stable_diffusion_v1_5/stable-diffusion-v1-5/snapshots/module'),
                        help='Path to pretrained SD model')
    parser.add_argument('--classifier_path',
                        type=str,
                        default="/media/inspur/disk/yychang_workspace/code/NoiseCLR/dataset/classifier_vangogh.pt",
                        help='Path to classifier model')
    parser.add_argument('--pretrained_mlp_path',
                        type=str,
                        default=os.path.join(root_path, 'code/NoiseCLR/mlp_BlackHair_0.3_ddp_1.pt'))
    parser.add_argument('--test_img_path',
                        type=str,
                        default='/media/inspur/disk/yychang_workspace/data/ffhq1024/00035.png',
                        help='a single img to be tested')
    parser.add_argument('--warmup_steps',
                        type=int,
                        default=10,
                        help='number of warmup steps')
    parser.add_argument('--lr_scheduler',
                        type=str,
                        default="constant",
                        help='学习率衰减策略')

    return parser.parse_args()
