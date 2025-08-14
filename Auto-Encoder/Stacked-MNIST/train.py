#!/usr/bin/env python3
"""
高精度端到端超球面自编码器（128 维潜码 + 标签监督）+ 测试集验证
加入固定线性分类器进行潜在空间评估
"""
import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from pathlib import Path
from lpips import LPIPS

# ---------- 全局种子 ----------
torch.manual_seed(2024)
np.random.seed(2024)

# ---------- 路径 ----------
ROOT       = Path("/root/autodl-tmp")
TRAIN_IMG  = ROOT / "mapper/Datasets/Stacked_MNIST/stacked_mnist/train/images.npy"
TRAIN_LBL  = ROOT / "mapper/Datasets/Stacked_MNIST/stacked_mnist/train/labels.npy"
TEST_IMG   = ROOT / "mapper/Datasets/Stacked_MNIST/stacked_mnist/test/images.npy"
TEST_LBL   = ROOT / "mapper/Datasets/Stacked_MNIST/stacked_mnist/test/labels.npy"
CKPT_DIR   = ROOT / "mapper/high_precision_ae_128"
os.makedirs(CKPT_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- 超参 ----------
BATCH      = 256
EPOCHS     = 400
LATENT     = 128
SAVE_EVERY = 10
LPIPS_W    = 0.2
CLS_W      = 0.1               # 分类损失权重

# ---------- 数据集 ----------
class StackedMNISTWithLabel(Dataset):
    def __init__(self, img_path, lbl_path):
        imgs = np.load(img_path).astype(np.float32) / 255.0
        lbls = np.load(lbl_path).astype(np.int64)
        self.images = np.transpose(imgs, (0, 3, 1, 2))  # (N,3,28,28)
        self.labels = lbls                              # (N,3)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx]), \
               torch.from_numpy(self.labels[idx])      # 返回 (3,) 标签

# ---------- DataLoader ----------
train_set = StackedMNISTWithLabel(TRAIN_IMG, TRAIN_LBL)
test_set  = StackedMNISTWithLabel(TEST_IMG, TEST_LBL)

train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=BATCH, shuffle=False,
                         num_workers=4, pin_memory=True)

# ---------- 网络 ----------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return F.relu(self.skip(x) + self.conv(x))

class Encoder(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(3, 64),   nn.AvgPool2d(2),   # 14
            ResidualBlock(64, 128), nn.AvgPool2d(2),   # 7
            ResidualBlock(128, 256), nn.AvgPool2d(2),  # 3
            nn.Flatten(),
            nn.Linear(256*3*3, 512),
            nn.ReLU(),
            nn.Linear(512, latent)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)   # 单位球面

class Decoder(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 3 * 3),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=1),  # 3→7
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 7→14
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 14→28
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 3, 3)
        return self.decoder(h)

encoder    = Encoder().to(device)
decoder    = Decoder().to(device)
classifier = nn.Linear(LATENT, 1000).to(device)   # 1000 = 10^3 类别

# ---------- 损失函数 ----------
mse_fn   = nn.MSELoss()
lpips_fn = LPIPS(net='vgg').to(device)
def loss_fn(rec, tgt):
    return mse_fn(rec, tgt) + LPIPS_W * lpips_fn(rec, tgt).mean()

opt      = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()),
                             lr=2e-4, weight_decay=1e-4)
opt_cls  = torch.optim.Adam(classifier.parameters(), lr=1e-3)
sched    = CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=1)
scaler   = GradScaler()
writer   = SummaryWriter(log_dir=CKPT_DIR / 'logs')

# ---------- 工具 ----------
def encode_label(lbl_tensor):
    return lbl_tensor[:, 0] * 100 + lbl_tensor[:, 1] * 10 + lbl_tensor[:, 2]

# ---------- 训练 ----------
best_loss = 1e9
for epoch in range(1, EPOCHS+1):
    encoder.train(); decoder.train(); classifier.train()
    running_mse, running_acc = 0., 0.
    for img, lbl in train_loader:
        img  = img.to(device, non_blocking=True)
        lbl_idx = encode_label(lbl).to(device)

        opt.zero_grad()
        opt_cls.zero_grad()
        with autocast():
            z = encoder(img)
            rec = decoder(z)
            recon_loss = loss_fn(rec, img)
            logits = classifier(z)
            cls_loss = F.cross_entropy(logits, lbl_idx)
            loss = recon_loss + CLS_W * cls_loss

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.step(opt_cls)
        scaler.update()
        sched.step()

        running_mse += recon_loss.item()
        running_acc += (logits.argmax(1) == lbl_idx).float().mean().item()

    avg_mse = running_mse / len(train_loader)
    avg_acc = running_acc / len(train_loader)
    print(f"[{epoch:03d}] Train MSE={avg_mse:.5f}  Acc={avg_acc:.4f}  LR={sched.get_last_lr()[0]:.2e}")
    writer.add_scalar('Loss/MSE', avg_mse, epoch)
    writer.add_scalar('Acc', avg_acc, epoch)

    # ---------- 测试集 ----------
    encoder.eval(); decoder.eval(); classifier.eval()
    test_mse, test_acc = 0., 0.
    with torch.no_grad():
        for img, lbl in test_loader:
            img  = img.to(device)
            lbl_idx = encode_label(lbl).to(device)
            z = encoder(img)
            rec = decoder(z)
            test_mse += loss_fn(rec, img).item()
            logits = classifier(z)
            test_acc += (logits.argmax(1) == lbl_idx).float().mean().item()

    avg_test_mse = test_mse / len(test_loader)
    avg_test_acc = test_acc / len(test_loader)
    print(f"   ↳ [Test]  MSE={avg_test_mse:.5f}  Acc={avg_test_acc:.4f}")
    writer.add_scalar('Val/MSE', avg_test_mse, epoch)
    writer.add_scalar('Val/Acc', avg_test_acc, epoch)

    # ---------- 保存图像和模型 ----------
    if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
        with torch.no_grad():
            sample, _ = next(iter(train_loader))
            sample = sample[:64].to(device)
            rec = decoder(encoder(sample))
            grid = torch.cat([sample, rec], 0)
            save_image(grid, CKPT_DIR / f'recon_{epoch:03d}.png', nrow=8)
        torch.save({'enc': encoder.state_dict(),
                    'dec': decoder.state_dict(),
                    'cls': classifier.state_dict()},
                   CKPT_DIR / f'ae_{epoch:03d}.pth')
        if avg_mse < best_loss:
            best_loss = avg_mse
            torch.save({'enc': encoder.state_dict(),
                        'dec': decoder.state_dict(),
                        'cls': classifier.state_dict()},
                       CKPT_DIR / 'ae_best.pth')

# ---------- 最终保存 ----------
torch.save({'enc': encoder.state_dict(),
            'dec': decoder.state_dict(),
            'cls': classifier.state_dict()},
           CKPT_DIR / 'ae_final.pth')
print("✅ 128维潜码 + 标签训练完成！")
writer.close()