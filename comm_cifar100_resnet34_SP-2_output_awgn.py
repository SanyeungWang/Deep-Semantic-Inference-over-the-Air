#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import argparse
import math
import time
import datetime
import logging
import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet34
from torchview import draw_graph

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--nc', type=int, default=1024, help='#channel uses (bottleneck dim)')
    p.add_argument('--snr', type=float, default=5, help='SNR in dB for AWGN layer')
    p.add_argument('--batch', type=int, default=128, help='mini-batch size')
    p.add_argument('--epochs', type=int, default=100, help='training epochs')
    p.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    p.add_argument('--workers', type=int, default=6, help='#dataloader workers')
    p.add_argument('--seed', type=int, default=42, help='random seed')
    return p.parse_args()


def init_logger(run_name, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.log")

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s,%(msecs)03d %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    logger.info(f"Logging to {log_path}")
    return logger


# Convert given SNR[dB] to SNR[linear] to sigma[standard deviation of AWGN noise]
def snr_db_to_sigma(snr_db):
    snr_linear = math.pow(10.0, snr_db / 10.0)
    return 1.0 / math.sqrt(snr_linear)


# Inject Additive White Gaussian Noise into the feature vector according to standard deviation sigma
class AWGN(nn.Module):
    def __init__(self, snr_db):
        super().__init__()
        self.sigma = snr_db_to_sigma(snr_db)

    def forward(self, x):
        noise = torch.randn_like(x, device=x.device) * self.sigma  # noise generation
        return x + noise


class ResNet34_CIFAR(nn.Module):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        # ---------- Stem ----------
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()

        # ---------- Layer1 : 3 × BasicBlock, 64 → 64 ----------
        # Block-1
        self.l1_b1_conv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b1_bn1 = nn.BatchNorm2d(64)
        self.l1_b1_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b1_bn2 = nn.BatchNorm2d(64)
        # Block-2
        self.l1_b2_conv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b2_bn1 = nn.BatchNorm2d(64)
        self.l1_b2_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b2_bn2 = nn.BatchNorm2d(64)
        # Block-3
        self.l1_b3_conv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b3_bn1 = nn.BatchNorm2d(64)
        self.l1_b3_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b3_bn2 = nn.BatchNorm2d(64)

        # ---------- Layer2 : 4 × BasicBlock, 64 → 128, stride=2 ----------
        self.ds2 = nn.Sequential(
            nn.Conv2d(64, 128, 1, 2, bias=False),
            nn.BatchNorm2d(128)
        )
        self.l2_b1_conv1 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)  # 32 → 16
        self.l2_b1_bn1 = nn.BatchNorm2d(128)
        self.l2_b1_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b1_bn2 = nn.BatchNorm2d(128)
        # Block-2
        self.l2_b2_conv1 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b2_bn1 = nn.BatchNorm2d(128)
        self.l2_b2_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b2_bn2 = nn.BatchNorm2d(128)
        # Block-3
        self.l2_b3_conv1 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b3_bn1 = nn.BatchNorm2d(128)
        self.l2_b3_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b3_bn2 = nn.BatchNorm2d(128)
        # Block-4
        self.l2_b4_conv1 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b4_bn1 = nn.BatchNorm2d(128)
        self.l2_b4_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b4_bn2 = nn.BatchNorm2d(128)

        # ---------- Layer3 : 6 × BasicBlock, 128 → 256, stride=2 ----------
        self.ds3 = nn.Sequential(
            nn.Conv2d(128, 256, 1, 2, bias=False),
            nn.BatchNorm2d(256)
        )
        self.l3_b1_conv1 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)  # 16 → 8
        self.l3_b1_bn1 = nn.BatchNorm2d(256)
        self.l3_b1_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b1_bn2 = nn.BatchNorm2d(256)
        # Block-2
        self.l3_b2_conv1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b2_bn1 = nn.BatchNorm2d(256)
        self.l3_b2_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b2_bn2 = nn.BatchNorm2d(256)
        # Block-3
        self.l3_b3_conv1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b3_bn1 = nn.BatchNorm2d(256)
        self.l3_b3_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b3_bn2 = nn.BatchNorm2d(256)
        # Block-4
        self.l3_b4_conv1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b4_bn1 = nn.BatchNorm2d(256)
        self.l3_b4_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b4_bn2 = nn.BatchNorm2d(256)
        # Block-5
        self.l3_b5_conv1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b5_bn1 = nn.BatchNorm2d(256)
        self.l3_b5_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b5_bn2 = nn.BatchNorm2d(256)
        # Block-6
        self.l3_b6_conv1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b6_bn1 = nn.BatchNorm2d(256)
        self.l3_b6_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b6_bn2 = nn.BatchNorm2d(256)

        # ---------- Layer4 : 3 × BasicBlock, 256 → 512, stride=2 ----------
        self.ds4 = nn.Sequential(
            nn.Conv2d(256, 512, 1, 2, bias=False),
            nn.BatchNorm2d(512)
        )
        self.l4_b1_conv1 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)  # 8 → 4
        self.l4_b1_bn1 = nn.BatchNorm2d(512)
        self.l4_b1_conv2 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.l4_b1_bn2 = nn.BatchNorm2d(512)
        # Block-2
        self.l4_b2_conv1 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.l4_b2_bn1 = nn.BatchNorm2d(512)
        self.l4_b2_conv2 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.l4_b2_bn2 = nn.BatchNorm2d(512)
        # Block-3
        self.l4_b3_conv1 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.l4_b3_bn1 = nn.BatchNorm2d(512)
        self.l4_b3_conv2 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.l4_b3_bn2 = nn.BatchNorm2d(512)

        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 512×1×1
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

        for name, mod in self.named_modules():
            if isinstance(mod, nn.BatchNorm2d) and name.endswith('bn2'):
                nn.init.constant_(mod.weight, 0)

    @staticmethod
    def _basic_block(x: torch.Tensor,
                     conv1: nn.Module, bn1: nn.Module,
                     conv2: nn.Module, bn2: nn.Module,
                     downsample: nn.Module = None) -> torch.Tensor:
        identity = x if downsample is None else downsample(x)
        out = F.relu(bn1(conv1(x)), inplace=True)
        out = bn2(conv2(out))
        return F.relu(out + identity, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- Stem ----
        x = self.conv1(x);
        x = self.bn1(x);
        x = self.relu(x)
        x = self.maxpool(x)  # Identity

        # ---- Layer1 ----
        x = self._basic_block(x, self.l1_b1_conv1, self.l1_b1_bn1,
                              self.l1_b1_conv2, self.l1_b1_bn2)
        x = self._basic_block(x, self.l1_b2_conv1, self.l1_b2_bn1,
                              self.l1_b2_conv2, self.l1_b2_bn2)
        x = self._basic_block(x, self.l1_b3_conv1, self.l1_b3_bn1,
                              self.l1_b3_conv2, self.l1_b3_bn2)

        # ---- Layer2 ----
        x = self._basic_block(x, self.l2_b1_conv1, self.l2_b1_bn1,
                              self.l2_b1_conv2, self.l2_b1_bn2,
                              downsample=self.ds2)
        x = self._basic_block(x, self.l2_b2_conv1, self.l2_b2_bn1,
                              self.l2_b2_conv2, self.l2_b2_bn2)
        x = self._basic_block(x, self.l2_b3_conv1, self.l2_b3_bn1,
                              self.l2_b3_conv2, self.l2_b3_bn2)
        x = self._basic_block(x, self.l2_b4_conv1, self.l2_b4_bn1,
                              self.l2_b4_conv2, self.l2_b4_bn2)

        # ---- Layer3 ----
        x = self._basic_block(x, self.l3_b1_conv1, self.l3_b1_bn1,
                              self.l3_b1_conv2, self.l3_b1_bn2,
                              downsample=self.ds3)
        x = self._basic_block(x, self.l3_b2_conv1, self.l3_b2_bn1,
                              self.l3_b2_conv2, self.l3_b2_bn2)
        x = self._basic_block(x, self.l3_b3_conv1, self.l3_b3_bn1,
                              self.l3_b3_conv2, self.l3_b3_bn2)
        x = self._basic_block(x, self.l3_b4_conv1, self.l3_b4_bn1,
                              self.l3_b4_conv2, self.l3_b4_bn2)
        x = self._basic_block(x, self.l3_b5_conv1, self.l3_b5_bn1,
                              self.l3_b5_conv2, self.l3_b5_bn2)
        x = self._basic_block(x, self.l3_b6_conv1, self.l3_b6_bn1,
                              self.l3_b6_conv2, self.l3_b6_bn2)

        # ---- Layer4 ----
        x = self._basic_block(x, self.l4_b1_conv1, self.l4_b1_bn1,
                              self.l4_b1_conv2, self.l4_b1_bn2,
                              downsample=self.ds4)
        x = self._basic_block(x, self.l4_b2_conv1, self.l4_b2_bn1,
                              self.l4_b2_conv2, self.l4_b2_bn2)
        x = self._basic_block(x, self.l4_b3_conv1, self.l4_b3_bn1,
                              self.l4_b3_conv2, self.l4_b3_bn2)

        # ---- Head ----
        x = self.avgpool(x).flatten(1)  # (B,512)
        return self.fc(x)  # (B,num_classes)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    args = parse_args()

    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_name = f"comm_cifar100_nc{args.nc}_snr{args.snr}"
    run_name = f"{base_name}_{ts}"
    logger = init_logger(run_name)

    args_line = ", ".join(f"{k}={v}" for k, v in vars(args).items())
    logger.info(f"Args: {args_line}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        logger.info(f'Using device {device} ({gpu_name})')
    else:
        logger.info('Using CPU')

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    transform = transforms.ToTensor()  # scales pixels to [0, 1]

    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    g = torch.Generator()
    g.manual_seed(args.seed)

    def _worker_init(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True, generator=g,
                                               num_workers=args.workers, pin_memory=True, worker_init_fn=_worker_init)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, generator=g,
                                              num_workers=args.workers, pin_memory=True, worker_init_fn=_worker_init)

    backbone = ResNet34_CIFAR(num_classes=100).to(device)

    class NoisyNet(nn.Module):
        def __init__(self, snr_db: float, backbone: nn.Module):
            super().__init__()
            self.backbone = backbone
            self.channel = AWGN(snr_db)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = self.backbone(x)  # (B,100)
            logits = F.normalize(logits, p=2.0, dim=1, eps=1e-8) * math.sqrt(logits.size(1))
            logits = self.channel(logits)
            return logits

    model = NoisyNet(args.snr, backbone).to(device)

    loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0)

    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        last_log_time = epoch_start
        running_loss = 0.0
        freq = max(1, len(train_loader) // 10)

        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            lr_now = opt.param_groups[0]['lr']

            if step % freq == 0 or step == len(train_loader):
                now = time.time()
                interval = now - last_log_time
                last_log_time = now

                logger.info(
                    f"Epoch:[{epoch}/{args.epochs}] "
                    f"Iter:[{step}/{len(train_loader)}] "
                    f"Time:{interval:.2f}s "
                    f"lr:{lr_now:.4e} "
                    f"Loss:{loss.item():.4f}"
                )

        scheduler.step()

        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)

        os.makedirs("weights", exist_ok=True)
        if test_acc > best_acc:
            best_acc = test_acc
            best_path = f'weights/{run_name}_best.pth'
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best model saved to {best_path} (acc={best_acc:.4f})")

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(train_loader)
        logger.info(
            f"Epoch:[{epoch}/{args.epochs}] "
            f"epoch_time:{epoch_time:.2f}s "
            f"avg_loss:{avg_loss:.4f} "
            f"train_acc:{train_acc:.4f} "
            f"test_acc:{test_acc:.4f} "
            f"best_acc:{best_acc:.4f}"
        )


if __name__ == '__main__':
    main()
