#!/usr/bin/env python3
import argparse, os, random
from pathlib import Path
import torch, pandas as pd
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from tqdm import tqdm

LAT_DIM = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 必须与训练脚本完全一致
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch)
        )
    def forward(self, x):
        return torch.relu(x + self.block(x))

class Decoder(nn.Module):
    def __init__(self, latent_dim=LAT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d(64,   32, 4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU(True),
            ResidualBlock(32),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 512, 4, 4)
        return self.layers(h)
# ----------------- 主流程 -----------------
def main(args):
    dec = Decoder().eval().to(device)
    dec.load_state_dict(torch.load(args.ckpt, map_location=device))

    # 潜码
    lat = pd.read_csv(args.csv, header=None).values.astype("float32")
    lat = torch.from_numpy(lat).to(device)

    # 用 ImageFolder 拿类别名
    split_dir = Path(args.data) / args.split
    ds = ImageFolder(split_dir)
    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}

    # 构造 类别 -> 行号列表
    from collections import defaultdict
    class_rows = defaultdict(list)
    for row_idx, (_, label) in enumerate(ds.samples):
        class_rows[idx_to_class[label]].append(row_idx)

    # 输出目录
    out_dir = Path(args.out) / args.split; out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for cls_name, rows in tqdm(class_rows.items(), desc="Classes"):
            # 随机采样
            sample_rows = random.sample(rows, min(len(rows), args.images_per_class))
            cls_dir = out_dir / cls_name
            cls_dir.mkdir(exist_ok=True)

            for start in range(0, len(sample_rows), args.batch):
                idxs = sample_rows[start: start + args.batch]
                imgs = dec(lat[idxs])
                for j, img in enumerate(imgs):
                    save_image(img, cls_dir / f"{start + j:04d}.png")

    print(f"✅ 重建完成，输出目录：{out_dir}")

# ----------------- 参数解析 -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   default="ckpt/val_latent_codes_best.csv")
    parser.add_argument("--ckpt",  default="/root/autodl-tmp/mapper-tinyImage/ckpt/best_dec.pth")
    parser.add_argument("--data",  default="/root/autodl-tmp/mapper-tinyImage/datasets")
    parser.add_argument("--out",   default="outputs/recon_by_class-64*64")
    parser.add_argument("--images_per_class", type=int, default=10)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    args = parser.parse_args()
    main(args)