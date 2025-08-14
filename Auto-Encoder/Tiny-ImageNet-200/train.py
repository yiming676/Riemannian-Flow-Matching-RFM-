#!/usr/bin/env python3
"""
tiny-ImageNet 64×64  自监督 AE + RegNet400MF
"""
import os, sys, time, math, logging
from pathlib import Path
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torchvision.models import regnet_x_400mf
from torchvision.utils import save_image
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from lpips import LPIPS
from kornia.losses import SSIMLoss

# ------------------- 日志 --------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ------------------- 超参 --------------------
DATA_DIR = Path("/root/autodl-tmp/mapper-tinyImage/datasets")
CKPT_DIR = Path("ckpt");
CKPT_DIR.mkdir(exist_ok=True)
IMG_SIZE = 64
LAT_DIM = 128
BATCH = 256 if torch.cuda.is_available() else 16
EPOCHS = 200
LR = 2e-5
PATIENCE = 15
WARM_STEPS = 100
CLIP_NORM = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- 数据 --------------------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
val_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_ds = datasets.ImageFolder((DATA_DIR / "train").as_posix(), train_tf)
val_ds = datasets.ImageFolder((DATA_DIR / "val").as_posix(), val_tf)
train_loader = DataLoader(train_ds, BATCH, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, BATCH, shuffle=False, num_workers=4, pin_memory=True)


# ------------------- 模型 --------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim, dropout=0.1):
        super().__init__()
        base = regnet_x_400mf(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        out_dim = base.fc.in_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        feat = self.pool(self.backbone(x)).flatten(1)
        z = self.fc(feat)
        return F.normalize(z, dim=1)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self


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
        return F.relu(x + self.block(x))


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            ResidualBlock(32),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 512, 4, 4)
        return self.layers(h)


# ------------------- 损失 --------------------
mse_fn = nn.MSELoss()
lpips_fn = LPIPS(net='vgg', verbose=False).to(device)
ssim_fn = SSIMLoss(window_size=11, max_val=1.0).to(device)


def loss_fn(rec, tgt, z):
    if torch.isnan(rec).any():
        raise RuntimeError("NaN in decoder output")
    mse = mse_fn(rec, tgt)
    lp = lpips_fn(rec, tgt).mean()
    ssim = 1 - torch.clamp(ssim_fn(rec, tgt), 0, 1)
    sphere = ((z.norm(dim=1) - 1) ** 2).mean()
    loss = mse + 0.8 * lp + 0.3 * ssim + 0.05 * sphere
    return mse, lp, ssim, sphere, loss


# ------------------- 工具函数 --------------------
@torch.no_grad()
def save_latent_codes(loader, filename):
    enc.eval()
    codes = []
    for x, _ in tqdm(loader, desc="Extract latent"):
        x = x.to(device)
        codes.append(enc(x).cpu().numpy())
    codes = np.concatenate(codes, 0).astype(np.float32)
    pd.DataFrame(codes).to_csv(CKPT_DIR / filename, index=False)
    logger.info(f"Saved {CKPT_DIR / filename}")


@torch.no_grad()
def reconstruct_with_tree(csv_path, ckpt_path, out_root, split="val"):
    os.makedirs(out_root, exist_ok=True)
    dec = Decoder().eval().to(device)
    dec.load_state_dict(torch.load(ckpt_path, map_location=device))

    codes = pd.read_csv(csv_path).values.astype("float32")
    codes = torch.from_numpy(codes).to(device)

    # 用 ImageFolder 拿类别名
    ds = datasets.ImageFolder((DATA_DIR / split).as_posix())
    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}

    # 类别 -> 行号
    class_rows = defaultdict(list)
    for i, (_, label) in enumerate(ds.samples):
        class_rows[idx_to_class[label]].append(i)

    # 按类别目录重建
    for cls_name, rows in tqdm(class_rows.items(), desc="Reconstruct"):
        cls_dir = Path(out_root) / cls_name
        cls_dir.mkdir(exist_ok=True)
        for start in range(0, len(rows), BATCH):
            idxs = rows[start: start + BATCH]
            imgs = dec(codes[idxs])
            for j, img in enumerate(imgs):
                save_image(img, cls_dir / f"{start + j:05d}.png")
    logger.info(f"Tree structure saved -> {out_root}")


@torch.no_grad()
def visualize_latents(csv_path, save_path="outputs/tSNE/tsne_latent.png"):
    codes = pd.read_csv(csv_path).values.astype(np.float32)
    z2d = TSNE(n_components=2, random_state=42).fit_transform(codes)
    plt.figure(figsize=(8, 8))
    plt.scatter(z2d[:, 0], z2d[:, 1], s=2, alpha=0.6)
    plt.title("t-SNE of latent codes")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"t-SNE saved -> {save_path}")


@torch.no_grad()
def visualize_3d_tsne(csv_path, save_html="outputs/tSNE/3d_tsne_tuned.html", n_samples=3000):
    import plotly.express as px

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    codes = pd.read_csv(csv_path).values.astype("float32")

    ds = ImageFolder((DATA_DIR / "val").as_posix())
    labels = [Path(p).parts[-2] for p, _ in ds.samples]

    if len(codes) > n_samples:
        idx = np.random.choice(len(codes), n_samples, replace=False)
        codes, labels = codes[idx], [labels[i] for i in idx]

    # 使用 t-SNE 进行降维
    tsne = TSNE(
        n_components=3,
        perplexity=10,
        early_exaggeration=32,
        learning_rate=800,
        init='pca',
        n_iter=2500,
        random_state=42
    )
    z3d = tsne.fit_transform(codes)

    # 创建 3D 散点图
    fig = px.scatter_3d(
        x=z3d[:, 0], y=z3d[:, 1], z=z3d[:, 2],
        color=labels,
        size_max=1.5,
        opacity=0.8,
        title="3D t-SNE (Enhanced Clustering)"
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[z3d[:, 0].min() * 1.2, z3d[:, 0].max() * 1.2]),
            yaxis=dict(range=[z3d[:, 1].min() * 1.2, z3d[:, 1].max() * 1.2]),
            zaxis=dict(range=[z3d[:, 2].min() * 1.2, z3d[:, 2].max() * 1.2]),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.8))
        )
    )
    fig.write_html(save_html)
    logger.info(f"Enhanced 3D t-SNE saved -> {save_html}")


@torch.no_grad()
def validate(epoch):
    enc.eval();
    dec.eval()
    totals = dict(loss=0., mse=0., lpips=0., ssim=0., sphere=0.)
    N = 0
    for x, _ in val_loader:
        x = x.to(device)
        z = enc(x)
        rec = dec(z)
        mse, lp, ssim, sphere, loss = loss_fn(rec, x, z)
        bs = x.size(0)
        for k, v in zip(totals.keys(), [loss, mse, lp, ssim, sphere]):
            totals[k] += v.item() * bs
        N += bs
    avg = {k: v / N for k, v in totals.items()}
    logger.info(f"Epoch {epoch:03d}  val_loss={avg['loss']:.4f}, "
                f"mse={avg['mse']:.4f}, lpips={avg['lpips']:.4f}, "
                f"ssim={avg['ssim']:.4f}, sphere={avg['sphere']:.4f}")
    return avg['loss']


# ------------------- 训练 --------------------
if __name__ == "__main__":
    enc = Encoder(LAT_DIM).to(device)
    dec = Decoder(LAT_DIM).to(device)

    opt = torch.optim.AdamW(chain(enc.parameters(), dec.parameters()),
                            lr=LR, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=len(train_loader) * EPOCHS)
    scaler = GradScaler()
    writer = SummaryWriter(CKPT_DIR / "logs")

    global_step = 0
    best_val = 1e9
    patience_cnt = 0

    for epoch in range(1, EPOCHS + 1):
        enc.train();
        dec.train()
        total = dict(loss=0., mse=0., lpips=0., ssim=0., sphere=0.)
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}")
        for x, _ in pbar:
            x = x.to(device)
            global_step += 1

            if global_step <= WARM_STEPS:
                lr_scale = min(1.0, global_step / WARM_STEPS)
                for pg in opt.param_groups:
                    pg["lr"] = LR * lr_scale

            opt.zero_grad()
            with autocast():
                z = enc(x)
                rec = dec(z)
                mse, lp, ssim, sphere, loss = loss_fn(rec, x, z)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                chain(enc.parameters(), dec.parameters()), CLIP_NORM)
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            for k, v in zip(total.keys(), [loss, mse, lp, ssim, sphere]):
                total[k] += v.item() * bs
            n += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = {k: v / n for k, v in total.items()}
        for k, v in avg.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)

        val_loss = validate(epoch)
        writer.add_scalar("val/loss", val_loss, epoch)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            patience_cnt = 0
            torch.save(enc.state_dict(), CKPT_DIR / "best_enc.pth")
            torch.save(dec.state_dict(), CKPT_DIR / "best_dec.pth")
            save_latent_codes(val_loader, "val_latent_codes_best.csv")
        else:
            patience_cnt += 1

        torch.save(enc.state_dict(), CKPT_DIR / "last_enc.pth")
        torch.save(dec.state_dict(), CKPT_DIR / "last_dec.pth")

        if patience_cnt >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    writer.close()
    reconstruct_with_tree(CKPT_DIR / "val_latent_codes_best.csv")
    visualize_3d_tsne(CKPT_DIR / "val_latent_codes_best.csv")
    visualize_latents(CKPT_DIR / "val_latent_codes_best.csv")