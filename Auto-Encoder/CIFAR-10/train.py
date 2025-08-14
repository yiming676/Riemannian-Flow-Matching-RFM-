#!/usr/bin/env python3
"""
CIFAR-10 → 128-dim unit-sphere latent code
可重建、可保存 csv、可继续训练（改resume）
"""
import os, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torchvision import datasets, transforms
from pathlib import Path
from lpips import LPIPS

# ---------- 全局 ----------
torch.manual_seed(2024)
np.random.seed(2024)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ---------- 路径 ----------
ROOT = Path("/root/autodl-tmp/mapper-CIFAR10")
DATA_DIR = ROOT / "datasets"
CKPT_DIR = ROOT / "ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- 超参 ----------
BATCH = 256
EPOCHS = 400
LATENT = 128
LR = 2e-4
SAVE_EVERY = 10
LPIPS_W = 0.2
CLS_W = 0.1

# ---------- 读取 CIFAR-10 (使用 torchvision) ----------
transform = transforms.ToTensor()  # 将数据转为 [0,1] 的 Tensor
train_ds = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
test_ds = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, BATCH, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_ds, BATCH, shuffle=False,
                         num_workers=4, pin_memory=True)


# ---------- 网络 (您的网络结构没有维度问题，无需修改) ----------
class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_out)
        )
        self.skip = nn.Conv2d(c_in, c_out, 1, bias=False) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return F.relu(self.skip(x) + self.net(x))


class Encoder(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(3, 64), nn.AvgPool2d(2),  # 16
            ResidualBlock(64, 128), nn.AvgPool2d(2),  # 8
            ResidualBlock(128, 256), nn.AvgPool2d(2),  # 4
            ResidualBlock(256, 512), nn.AvgPool2d(2),  # 2
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512), nn.ReLU(inplace=True),
            nn.Linear(512, latent)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class Decoder(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512 * 2 * 2),
            nn.ReLU(inplace=True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 2→4
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4→8
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8→16
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16→32
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 512, 2, 2)
        return self.deconv(h)


# ---------- 模型、损失函数、优化器 ----------
encoder = Encoder().to(device)
decoder = Decoder().to(device)
classifier = nn.Linear(LATENT, 10).to(device)

mse_fn = nn.MSELoss()
lpips_fn = LPIPS(net='vgg').to(device)


def loss_fn(rec, tgt):
    # LPIPS 期望输入在 [-1, 1]
    rec_lpips = rec * 2 - 1
    tgt_lpips = tgt * 2 - 1
    return mse_fn(rec, tgt) + LPIPS_W * lpips_fn(rec_lpips, tgt_lpips).mean()


# 【修正】将所有参数合并到一个优化器
params_to_optimize = list(encoder.parameters()) + list(decoder.parameters()) + list(classifier.parameters())
opt = torch.optim.AdamW(params_to_optimize, lr=LR, weight_decay=1e-4)
sched = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)  # T_0 设为10个 epochs，每 2*T_0 后周期加倍
scaler = GradScaler()
writer = SummaryWriter(CKPT_DIR / 'logs')

# ---------- 训练 ----------
best_loss = 1e9
start_time = time.time()
for epoch in range(1, EPOCHS + 1):
    encoder.train();
    decoder.train();
    classifier.train()
    run_recon_loss, run_acc = 0., 0.

    for img, lbl in train_loader:
        img, lbl = img.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
        opt.zero_grad()

        with autocast():
            z = encoder(img)
            rec = decoder(z)
            logits = classifier(z)

            recon_loss = loss_fn(rec, img)
            cls_loss = F.cross_entropy(logits, lbl)
            loss = recon_loss + CLS_W * cls_loss

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        # 【修正】使用保存的变量计算指标，提高效率
        run_recon_loss += recon_loss.item()
        run_acc += (logits.detach().argmax(1) == lbl).float().mean().item()

    avg_recon_loss = run_recon_loss / len(train_loader)
    avg_acc = run_acc / len(train_loader)
    print(
        f"[{epoch:03d}/{EPOCHS}] Train ReconLoss={avg_recon_loss:.5f}  Acc={avg_acc:.4f}  LR={sched.get_last_lr()[0]:.2e}")
    writer.add_scalar('Loss/Train_Recon', avg_recon_loss, epoch)
    writer.add_scalar('Acc/Train', avg_acc, epoch)

    # ---------- 验证 ----------
    encoder.eval();
    decoder.eval();
    classifier.eval()
    test_recon_loss, test_acc = 0., 0.
    with torch.no_grad():
        for img, lbl in test_loader:
            img, lbl = img.to(device), lbl.to(device)
            z = encoder(img)
            rec = decoder(z)
            logits = classifier(z)

            test_recon_loss += loss_fn(rec, img).item()
            test_acc += (logits.argmax(1) == lbl).float().mean().item()

    avg_test_recon_loss = test_recon_loss / len(test_loader)
    avg_test_acc = test_acc / len(test_loader)
    print(f"    ↳ [Test]  ReconLoss={avg_test_recon_loss:.5f}  Acc={avg_test_acc:.4f}")
    writer.add_scalar('Loss/Val_Recon', avg_test_recon_loss, epoch)
    writer.add_scalar('Acc/Val', avg_test_acc, epoch)

    # 【修正】学习率调度器在每个 epoch 后更新
    sched.step()

    # ---------- 保存图像 & 模型 ----------
    states = {
        'enc': encoder.state_dict(),
        'dec': decoder.state_dict(),
        'cls': classifier.state_dict(),
        'opt': opt.state_dict()
    }
    if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
        with torch.no_grad():
            sample, _ = next(iter(test_loader))  # 从测试集取样更具代表性
            sample = sample[:32].to(device)
            rec = decoder(encoder(sample))
            grid = torch.cat([sample, rec], 0)
            save_image(grid, CKPT_DIR / f'recon_{epoch:03d}.png', nrow=8)

        torch.save(states, CKPT_DIR / f'ae_{epoch:03d}.pth')

    # 【修正】基于验证集损失保存最佳模型
    if avg_test_recon_loss < best_loss:
        best_loss = avg_test_recon_loss
        print(f"🎉 New best model saved with Val ReconLoss: {best_loss:.5f}")
        torch.save(states, CKPT_DIR / 'ae_best.pth')

# ---------- 最终模型 ----------
torch.save(states, CKPT_DIR / 'ae_final.pth')


# ---------- 导出 128 维潜码 csv ----------
@torch.no_grad()
def export_latents(model_path, dataset, split_name):
    print(f"\nExporting latents for {split_name} split...")
    states = torch.load(model_path, map_location=device)
    encoder.load_state_dict(states['enc'])
    encoder.eval()

    # 使用更大的 batch size 来加速导出
    data_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

    all_latents, all_labels = [], []
    for img, lbl in data_loader:
        img = img.to(device)
        all_latents.append(encoder(img).cpu().numpy())
        all_labels.append(lbl.numpy())

    all_latents = np.concatenate(all_latents, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 使用 Pandas 保存为带表头的 CSV
    df_latents = pd.DataFrame(all_latents, columns=[f'dim_{i}' for i in range(LATENT)])
    df_latents['label'] = all_labels

    output_path = CKPT_DIR / f"cifar10_{split_name}_latents.csv"
    df_latents.to_csv(output_path, index=False, float_format="%.6f")
    print(f"✅ CSV for {split_name} exported to {output_path}")
    print(f"   Latents shape: {all_latents.shape}, Labels shape: {all_labels.shape}")


# 使用最佳模型导出训练集和测试集的潜码
export_latents(CKPT_DIR / 'ae_best.pth', train_ds, 'train')
export_latents(CKPT_DIR / 'ae_best.pth', test_ds, 'test')

writer.close()
total_time = time.time() - start_time
print(f"\n🎉 训练完成！总耗时: {total_time / 3600:.2f} 小时")