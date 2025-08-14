import os
import numpy as np
import pandas as pd
import torch
from torchvision.utils import save_image
from pathlib import Path

# 从 model.py 导入 Decoder 类
from model import Decoder, LATENT

# ---------- 配置 ----------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64  # 一次重建并保存的图像数量，可根据显存调整
WEIGHTS_PATH = Path("/root/autodl-tmp/mapper-CIFAR10/ckpt/ae_best.pth")
INPUT_CSV = Path("/root/autodl-tmp/riemannian-fm-main/hyper_s128.csv")
OUTPUT_DIR = Path("/root/autodl-tmp/mapper-CIFAR10/recon01")


def main():
    print(f"Using device: {DEVICE}")

    # 1. 确保输出目录存在
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output images will be saved to: {OUTPUT_DIR}")

    # 2. 初始化模型并加载权重
    print(f"Loading decoder weights from {WEIGHTS_PATH}...")
    decoder = Decoder(latent=LATENT).to(DEVICE)

    if not WEIGHTS_PATH.exists():
        print(f"Error: Weights file not found at {WEIGHTS_PATH}")
        return

    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    decoder.load_state_dict(state_dict['dec'])
    decoder.eval()  # 设置为评估模式

    # 3. 读取CSV文件
    print(f"Reading latent vectors from {INPUT_CSV}...")
    if not INPUT_CSV.exists():
        print(f"Error: Input CSV file not found at {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)

    # 提取潜在向量列 (假设列名为 dim_0, dim_1, ...)
    latent_columns = [f'dim_{i}' for i in range(LATENT)]
    latents_np = df[latent_columns].values

    # 转换为 PyTorch Tensor
    latents_tensor = torch.from_numpy(latents_np).float().to(DEVICE)
    print(f"Loaded {len(latents_tensor)} latent vectors.")

    # 4. 分批解码并保存图像
    print("Reconstructing images from latent vectors...")
    num_batches = (len(latents_tensor) + BATCH_SIZE - 1) // BATCH_SIZE

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(latents_tensor))

            batch_latents = latents_tensor[start_idx:end_idx]

            # 解码
            recon_images = decoder(batch_latents)

            # 逐个保存图像
            for j in range(recon_images.size(0)):
                image_index = start_idx + j
                output_path = OUTPUT_DIR / f"recon_{image_index:05d}.png"
                save_image(recon_images[j], output_path)

            print(f"  Saved batch {i + 1}/{num_batches} (images {start_idx} to {end_idx - 1})")

    print("✅ Image reconstruction complete!")


if __name__ == '__main__':
    main()