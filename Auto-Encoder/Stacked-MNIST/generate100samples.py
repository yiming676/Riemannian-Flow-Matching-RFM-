#!/usr/bin/env python3
"""
随机生成 100 张 3×28×28 堆叠 MNIST 图像
"""
import torch, os
from torchvision.utils import save_image
from pathlib import Path

# -------------------- 路径 --------------------
CKPT_PATH = Path("/root/autodl-tmp/mapper/high_precision_ae_final/ae_best.pth")
OUT_DIR   = Path("/root/autodl-tmp/mapper/high_precision_ae_final/random_100")
os.makedirs(OUT_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------- 网络（与训练一致） --------------------
import torch.nn as nn, torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x): return F.relu(self.skip(x) + self.conv(x))

class Decoder(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent, 512), nn.ReLU(),
            nn.Linear(512, 256 * 3 * 3), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 2, 1),  # 3→7
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 7→14
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 14→28
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 3, 1), nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 3, 3)
        return self.decoder(h)

decoder = Decoder().to(device)
ckpt = torch.load(CKPT_PATH, map_location=device)
decoder.load_state_dict(ckpt['dec'])
decoder.eval()

# -------------------- 随机生成 --------------------
N = 100
with torch.no_grad():
    z = torch.randn(N, 64, device=device)
    z = F.normalize(z, dim=1)          # 保证单位球面
    imgs = decoder(z)                  # [100,3,28,28]

# -------------------- 保存 --------------------
for i, img in enumerate(imgs):
    save_image(img, OUT_DIR / f"recon_{i:05d}.png")

save_image(imgs, OUT_DIR / "grid_10x10.png", nrow=10)

print(f"✅ 已生成 100 张随机堆叠 MNIST → {OUT_DIR}")