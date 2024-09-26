from __future__ import print_function
import os
import gc
import argparse
from pickle import FALSE, TRUE, load
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data import MedShapeNet
import numpy as np
from torch.utils.data import DataLoader
from model import DIT
import os
import shutil


def _init_(args):
    """Initialize directories for checkpoints and backup source files."""
    checkpoints_dir = 'checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)
    exp_dir = os.path.join(checkpoints_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    backup_files = ['main.py', 'model.py', 'data.py']
    for file in backup_files:
        src_path = file
        dst_path = os.path.join(exp_dir, f"{file}.backup")
        shutil.copy(src_path, dst_path)
    print(f"Initialized directories and backed up files to: {exp_dir}")

def train(args, net, train_loader, test_loader):
    """Train and evaluate the model."""
    print("Using AdamW optimizer")
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    epoch_factor = args.epochs / 100.0
    scheduler = MultiStepLR(optimizer,
                            milestones=[int(30 * epoch_factor), int(60 * epoch_factor), int(80 * epoch_factor)],
                            gamma=0.1)

    best_test_info = None
    if args.load_model:
        print(f'Loading parameters from {args.model_path}')
        net.load_state_dict(torch.load(args.model_path))
        net.eval()
    mode = 'Testing' if args.eval else 'Training'
    print(mode)
    for epoch in range(args.epochs):
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch: {epoch}, LR: {current_lr:.6f}")
        if not args.eval:
            train_info = net._train_one_epoch(epoch=epoch, train_loader=train_loader, optimizer=optimizer, args=args)
            gc.collect()

        test_info = net._test_one_epoch(epoch=epoch, test_loader=test_loader, vis=args.vis)

        if not args.eval:
            scheduler.step()
        if best_test_info is None or best_test_info['loss'] < test_info['loss']:
            best_test_info = test_info
            best_test_info['stage'] = 'best_test'
            if not args.eval:
                torch.save(net.state_dict(), 'models/model.best.pth')
        if not args.eval:
            net.logger.write(best_test_info)
            torch.save(net.state_dict(), f'models/model.{epoch}.pth')
        else:
            break
        gc.collect()
    print('Final Mode:', mode)


class TrainingConfig:
    n_emb_dims: int = 256
    n_blocks: int = 12
    n_heads: int = 4
    n_iters: int = 2
    discount_factor: float = 0.9
    n_ff_dims: int = 1024
    temp_factor: float = 100
    dropout: float = 0.4
    lr: float = 0.00003
    momentum: float = 0.9
    seed: int = 4327
    eval: bool = False
    vis: bool = False
    cycle_consistency_loss: float = 0.1
    discrimination_loss: float = 0.5
    gaussian_noise: float = 0.05
    unseen: bool = True
    rot_factor: float = 4
    model_path: str = 'models/model.pth'
    batch_size: int = 2
    test_batch_size: int = 2
    epochs: int = 1
    n_points: int = 1000
    n_subsampled_points: int = 800
    corres_mode: bool = False
    GMCCE_Sharp: float = 30
    GMCCE_Thres: float = 0.6
    load_model: bool = False
    token_dim: int = 64

def parse_args():
    parser = argparse.ArgumentParser(description='PER')
    for field in TrainingConfig.__annotations__:
        default_value = getattr(TrainingConfig, field)
        parser.add_argument(f'--{field}', type=type(default_value), default=default_value)

    return parser.parse_args()


def main():
    args = parse_args()  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    print(args)
    train_loader = DataLoader(
        MedShapeNet(num_points=args.n_points,
                    num_subsampled_points=args.n_subsampled_points,
                    partition='train', gaussian_noise=args.gaussian_noise,
                    unseen=args.unseen, rot_factor=args.rot_factor),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0
    )

    test_loader = DataLoader(
        MedShapeNet(num_points=args.n_points,
                    num_subsampled_points=args.n_subsampled_points,
                    partition='test', gaussian_noise=args.gaussian_noise,
                    unseen=args.unseen, rot_factor=args.rot_factor),
        batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=0
    )

    net = DIT(args).cuda()

    if args.load_model:
        model_path = args.model_path if args.model_path != 0 else 'models/good.pth'
        if not os.path.exists(model_path):
            print("Can't find pretrained model")
            return
        net.load_state_dict(torch.load(model_path))
        net.eval()  
    train(args, net, train_loader, test_loader)
    print('OK')
if __name__ == '__main__':
    main()
