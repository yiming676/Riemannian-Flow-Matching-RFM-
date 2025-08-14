#!/usr/bin/env python3
"""
decode_128_to_stackedmnist.py
将S128维超球面的 .csv 数据转化为堆叠MNIST的 .npy文件，并展示几张重建图像。
指令：python decode_128_to_stackedmnist.py generated_s128_01.csv
"""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torchvision.utils import save_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load(r"E:\科研\浙大计算机\夏令营实习\mapper\high_precision_ae_128\ae_best.pth", map_location=device)

# ---------- 网络 ----------
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
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   nn.BatchNorm2d(32), nn.ReLU(True),   # 28
            nn.Conv2d(32, 3, 1), nn.Sigmoid()
        )
    def forward(self, z):
        h = self.fc(z).view(-1, 256, 3, 3)
        return self.decoder(h)

decoder = Decoder().to(device)
decoder.load_state_dict(ckpt['dec'])
decoder.eval()

# ---------- 主流程 ----------
@torch.no_grad()
def main(csv_path, out_dir=Path("./recon")):
    out_dir.mkdir(exist_ok=True, parents=True)
    # 1) 读取 CSV
    z_np = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)
    z = torch.from_numpy(z_np).to(device)

    # 2) 重建
    imgs = decoder(z)                      # (N,3,28,28)  [0,1]
    imgs_np = imgs.cpu().numpy().transpose(0, 2, 3, 1)   # (N,28,28,3)

    # 3) 保存 .npy
    np.save(out_dir / "reconstructed_s128_01.npy", imgs_np.astype(np.float32))
    print(f"已保存 {out_dir / 'reconstructed_s128_01.npy'}  形状 {imgs_np.shape}")

    # 4) 可视化 8×8
    viz_n = min(len(imgs), 64)
    grid = imgs[:viz_n]
    save_image(grid, out_dir / "recon_grid_s128_01.png", nrow=8)
    print(f"预览图 {out_dir / 'recon_grid_s128_01.png'}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python decode_128_to_stackedmnist.py <latent_128.csv>")
        sys.exit(1)
    main(sys.argv[1])
    