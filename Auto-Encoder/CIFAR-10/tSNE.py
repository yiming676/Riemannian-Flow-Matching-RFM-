#!/usr/bin/env python3

import os, time, logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- 路径 / 参数 ----------------
ROOT     = Path("/root/autodl-tmp/mapper-CIFAR10")
CKPT_DIR = ROOT / "ckpt"
DATA_DIR = ROOT / "datasets"
SAVE_DIR = ROOT / "tSNE_vis"; SAVE_DIR.mkdir(parents=True, exist_ok=True)

LATENT      = 128
BATCH       = 512
PERPLEXITY  = 15
RANDOM_SEED = 42

# ---------------- 数据集 ----------------
transform = transforms.ToTensor()   # [0,1]
test_ds  = datasets.CIFAR10(root=DATA_DIR, train=False,
                            download=True, transform=transform)
test_loader = DataLoader(test_ds, BATCH, shuffle=False,
                         num_workers=8, pin_memory=True)

# ---------------- 网络 ----------------
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
            ResidualBlock(3, 64),   nn.AvgPool2d(2),
            ResidualBlock(64, 128), nn.AvgPool2d(2),
            ResidualBlock(128, 256), nn.AvgPool2d(2),
            ResidualBlock(256, 512), nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(512*2*2, 512), nn.ReLU(inplace=True),
            nn.Linear(512, latent)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

# ---------------- 加载 encoder ----------------
encoder = Encoder().to(device)
encoder.load_state_dict(torch.load(CKPT_DIR / "ae_best.pth",
                                   map_location=device)["enc"])
encoder.eval()

# ---------------- 提取潜向量 + 标签 ----------------
@torch.no_grad()
def extract(loader):
    zs, ys = [], []
    for img, lbl in tqdm(loader, desc="Extract"):
        img = img.to(device)
        zs.append(encoder(img).cpu().numpy())
        ys.append(lbl.numpy())
    return np.concatenate(zs), np.concatenate(ys)

z_all, y_all = extract(test_loader)
class_names  = test_ds.classes  # ['plane','car',...]

# ---------------- 2-D t-SNE ----------------
logging.info("Running 2-D t-SNE …")
z2d = TSNE(n_components=2, perplexity=PERPLEXITY,
           random_state=RANDOM_SEED, n_iter=1000).fit_transform(z_all)

cmap = plt.cm.get_cmap("tab10", 10)
plt.figure(figsize=(8, 8))
scatter = plt.scatter(z2d[:, 0], z2d[:, 1],
                      c=y_all, cmap=cmap, s=8, alpha=0.8)
plt.legend(handles=scatter.legend_elements()[0], labels=class_names,
           loc='upper right', fontsize=8)
plt.title("CIFAR-10 2-D t-SNE")
plt.tight_layout()
plt.savefig(SAVE_DIR / "cifar10_test_2d.png", dpi=300)
plt.close()
logging.info("2-D t-SNE saved -> %s", SAVE_DIR / "cifar10_test_2d.png")

# ---------------- 3-D t-SNE ----------------
logging.info("Running 3-D t-SNE …")
z3d = TSNE(n_components=3, perplexity=PERPLEXITY,
           random_state=RANDOM_SEED, n_iter=1000).fit_transform(z_all)
# ---------- 3-D t-SNE（降采样 + 调透明度 + 视角） ----------
N_DISPLAY = 3000
if z3d.shape[0] > N_DISPLAY:
    idx = np.random.choice(z3d.shape[0], N_DISPLAY, replace=False)
    z3d_plot  = z3d[idx]
    labels_plot = [class_names[i] for i in y_all[idx]]
else:
    z3d_plot  = z3d
    labels_plot = [class_names[i] for i in y_all]

df = pd.DataFrame(dict(x=z3d_plot[:, 0],
                       y=z3d_plot[:, 1],
                       z=z3d_plot[:, 2],
                       label=labels_plot))

fig = px.scatter_3d(df, x="x", y="y", z="z",
                    color="label",
                    title=f"CIFAR-10 3-D t-SNE (random {N_DISPLAY} pts)",
                    opacity=0.5,
                    size_max=2)


fig.update_layout(
    scene=dict(
        xaxis=dict(showbackground=False, showticklabels=True,
                   gridcolor='rgba(200,200,200,0.3)', zeroline=False),
        yaxis=dict(showbackground=False, showticklabels=True,
                   gridcolor='rgba(200,200,200,0.3)', zeroline=False),
        zaxis=dict(showbackground=False, showticklabels=True,
                   gridcolor='rgba(200,200,200,0.3)', zeroline=False),
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.6))
    )
)

html_path = SAVE_DIR / "cifar10_test_3d_clean.html"
fig.write_html(html_path)
logging.info("clean 3-D t-SNE saved -> %s", html_path)

# ---------------- 潜向量 CSV ----------------
df_lat = pd.DataFrame(z_all, columns=[f"dim_{i}" for i in range(LATENT)])
df_lat["label"] = y_all
df_lat.to_csv(SAVE_DIR / "cifar10_test_latents.csv", index=False, float_format="%.6f")
logging.info("Latents CSV saved -> %s", SAVE_DIR / "cifar10_test_latents.csv")