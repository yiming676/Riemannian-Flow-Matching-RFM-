#!/usr/bin/env python3
"""
把 Tiny-ImageNet-200 全部图像编码成 latents.csv
"""
import torch, pandas as pd, numpy as np
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

LAT_DIM = 128
class Encoder(torch.nn.Module):
    def __init__(self, latent_dim=LAT_DIM):
        super().__init__()
        from torchvision.models import regnet_x_400mf
        base = regnet_x_400mf(pretrained=True)
        self.backbone = torch.nn.Sequential(*list(base.children())[:-2])
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(400, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, latent_dim)
        )
    def forward(self, x):
        feat = self.pool(self.backbone(x)).flatten(1)
        z = self.fc(feat)
        return torch.nn.functional.normalize(z, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = Encoder().eval().to(device)
enc.load_state_dict(torch.load("ckpt/best_enc.pth", map_location=device))

tf = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

from torchvision.datasets import ImageFolder
root = Path("/root/autodl-tmp/mapper-tinyImage/datasets")
all_ds = ImageFolder(root / "train", tf) + ImageFolder(root / "val", tf)
loader = DataLoader(all_ds, batch_size=512, shuffle=False,
                    num_workers=12, pin_memory=True)

latents = []
for x, _ in tqdm(loader, desc="Encoding"):
    x = x.to(device, non_blocking=True)
    with torch.no_grad():
        latents.append(enc(x).cpu().numpy())

latents = np.vstack(latents)
assert (np.abs(np.linalg.norm(latents, axis=1) - 1) < 1e-4).all()
pd.DataFrame(latents).to_csv("outputs/latents_all.csv",
                             index=False, header=False, float_format="%.6f")
print(f"✅ 完成，共 {len(latents)} 行")