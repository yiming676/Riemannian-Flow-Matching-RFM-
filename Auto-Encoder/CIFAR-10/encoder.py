# encoder.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

# 从 model.py 导入 Encoder 类
from model import Encoder, LATENT

# ---------- 配置 ----------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 512  # 可以根据您的显存调整
WEIGHTS_PATH = Path("/root/autodl-tmp/mapper-CIFAR10/ckpt/ae_best.pth")
DATA_DIR = Path("/root/autodl-tmp/mapper-CIFAR10/datasets")
OUTPUT_CSV = Path("/root/autodl-tmp/mapper-CIFAR10/ckpt/cifar10_train_latents.csv")


def main():
    print(f"Using device: {DEVICE}")

    # 1. 加载数据集
    print("Loading CIFAR-10 training data...")
    transform = transforms.ToTensor()
    train_ds = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. 初始化模型并加载权重
    print(f"Loading encoder weights from {WEIGHTS_PATH}...")
    encoder = Encoder(latent=LATENT).to(DEVICE)

    if not WEIGHTS_PATH.exists():
        print(f"Error: Weights file not found at {WEIGHTS_PATH}")
        return

    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    encoder.load_state_dict(state_dict['enc'])
    encoder.eval()  # 设置为评估模式

    # 3. 编码并收集结果
    print("Encoding images to latent vectors...")
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            latents = encoder(images)
            all_latents.append(latents.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            print(f"  Processed batch {i + 1}/{len(train_loader)}")

    # 4. 整合数据并保存为CSV
    print("Concatenating results...")
    final_latents = np.concatenate(all_latents, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)

    print(f"Final latents shape: {final_latents.shape}")
    print(f"Final labels shape: {final_labels.shape}")

    # 使用 Pandas 创建 DataFrame 并保存
    df = pd.DataFrame(final_latents, columns=[f'dim_{i}' for i in range(LATENT)])
    df['label'] = final_labels

    print(f"Saving CSV to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False, float_format="%.8f")

    print("✅ Encoding complete and CSV file saved successfully!")


if __name__ == '__main__':
    main()