#!/usr/bin/env python3
"""
生成新的 Stacked-MNIST 并对比重建差异
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image
from lpips import LPIPS
from PIL import Image

# ---------- 全局 ----------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = Path("/root/autodl-tmp/mapper/high_precision_ae_128/ae_best.pth")
out_dir   = Path("./high_precision_ae_128/eval_out")
out_dir.mkdir(exist_ok=True, parents=True)

# ---------- 复用原网络定义 ----------
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

    def forward(self, x):
        return nn.functional.relu(self.skip(x) + self.conv(x))

class Encoder(nn.Module):
    def __init__(self, latent=128):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(3, 64), nn.AvgPool2d(2),      # 14
            ResidualBlock(64, 128), nn.AvgPool2d(2),    # 7
            ResidualBlock(128, 256), nn.AvgPool2d(2),   # 3
            nn.Flatten(),
            nn.Linear(256*3*3, 512), nn.ReLU(),
            nn.Linear(512, latent)
        )

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=1)

class Decoder(nn.Module):
    def __init__(self, latent=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent, 512), nn.ReLU(),
            nn.Linear(512, 256*3*3), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),  # 7
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.BatchNorm2d(64), nn.ReLU(True),   # 14
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   nn.BatchNorm2d(32), nn.ReLU(True),    # 28
            nn.Conv2d(32, 3, 1), nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 3, 3)
        return self.decoder(h)

# ---------- 生成新 Stacked-MNIST ----------
def generate_stacked_mnist(n_samples):
    """返回 (n,3,28,28) 图像 + (n,3) 标签"""
    from torchvision.datasets import MNIST
    import torchvision.transforms as T

    mnist = MNIST(root="/tmp", train=False, download=True,
                  transform=T.ToTensor())
    imgs, labels = [], []
    for _ in range(n_samples):
        idx = [random.randint(0, len(mnist)-1) for _ in range(3)]
        stacked = torch.cat([mnist[i][0] for i in idx], dim=0)   # (3,28,28)
        label   = torch.tensor([mnist[i][1] for i in idx])
        imgs.append(stacked)
        labels.append(label)
    return torch.stack(imgs), torch.stack(labels)

# ---------- 推理 ----------
@torch.no_grad()
def main(n_samples=512):
    # 1) 生成新数据
    images, labels = generate_stacked_mnist(n_samples)
    images = images.to(device)          # (N,3,28,28)

    # 2) 加载权重
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    encoder.load_state_dict(ckpt['enc'])
    decoder.load_state_dict(ckpt['dec'])
    encoder.eval(); decoder.eval()

    # 3) 128 维潜码 + 重建
    z   = encoder(images)
    rec = decoder(z)

    # 4) 计算指标
    lpips_fn = LPIPS(net='vgg').to(device)
    mse  = nn.functional.mse_loss(rec, images).item()
    lpips= lpips_fn(rec, images).mean().item()

    psnrs, ssims = [], []
    images_np = images.cpu().numpy().transpose(0,2,3,1)
    rec_np    = rec.cpu().numpy().transpose(0,2,3,1)
    for i in range(n_samples):
        psnrs.append(psnr(images_np[i], rec_np[i], data_range=1.0))
        ssims.append(ssim(images_np[i], rec_np[i], data_range=1.0, channel_axis=2))
        # 4) 计算指标 ...
    metrics = {
        "MSE":  mse,
        "LPIPS": lpips,
        "PSNR":  np.mean(psnrs),
        "SSIM":  np.mean(ssims)
    }
    metrics = {k: float(v) for k, v in metrics.items()}

    print("差异指标：", metrics)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 5) 可视化 8×8 对比
    viz_n = min(n_samples, 64)
    orig_viz = images[:viz_n]
    rec_viz  = rec[:viz_n]
    grid = torch.cat([orig_viz, rec_viz], dim=0)   # (128,3,28,28)
    save_image(grid, out_dir / "recon_comparison.png", nrow=8)

    # 6) 保存潜向量
    torch.save(z.cpu(), out_dir / "latent_codes.pt")
    torch.save(labels, out_dir / "labels.pt")
    print("结果已保存至", out_dir)

if __name__ == "__main__":
    main(512)