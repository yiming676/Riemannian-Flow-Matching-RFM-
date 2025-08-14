#!/usr/bin/env python3
"""
embed_2_csv.py
把 Stacked-MNIST 的 images.npy → 128 维潜向量 csv
指令：python embed_2_csv.py /path/images.npy output.csv
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = Path("/root/autodl-tmp/mapper/high_precision_ae_128/ae_best.pth")

# ---------- 网络（同仓库） ----------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
    def forward(self, x): return F.relu(self.skip(x) + self.conv(x))

class Encoder(nn.Module):
    def __init__(self, latent=128):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(3, 64), nn.AvgPool2d(2),
            ResidualBlock(64, 128), nn.AvgPool2d(2),
            ResidualBlock(128, 256), nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(256*3*3, 512), nn.ReLU(),
            nn.Linear(512, latent)
        )
    def forward(self, x): return F.normalize(self.net(x), dim=1)

# ---------- 主流程 ----------
def embed_npy_to_csv(npy_path: str, csv_path: str, batch_size=256):
    # 1) 加载权重
    ckpt = torch.load(str(ckpt_path), map_location=device)
    encoder = Encoder().to(device)
    encoder.load_state_dict(ckpt['enc'])
    encoder.eval()

    # 2) 读取 .npy 并统一形状 (N,3,28,28)
    imgs = np.load(npy_path)
    if imgs.dtype != np.float32: imgs = imgs.astype(np.float32)
    if imgs.max() > 1.0: imgs /= 255.0
    if imgs.shape[-1] == 3: imgs = imgs.transpose(0, 3, 1, 2)  # NHWC→NCHW
    imgs = torch.from_numpy(imgs).to(device)

    # 3) 推理（分批避免显存爆）
    latents = []
    for start in range(0, len(imgs), batch_size):
        end = min(start + batch_size, len(imgs))
        z = encoder(imgs[start:end])
        latents.append(z.detach().cpu().numpy())
    latents = np.concatenate(latents, axis=0)

    # 4) 保存 CSV
    np.savetxt(csv_path, latents, delimiter=',', fmt='%.6f')
    print(f"已生成 {csv_path}  形状 {latents.shape}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python embed_2_csv.py <images.npy> <output.csv>")
        sys.exit(1)
    embed_npy_to_csv(sys.argv[1], sys.argv[2])