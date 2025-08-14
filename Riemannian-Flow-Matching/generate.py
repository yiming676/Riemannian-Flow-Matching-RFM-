"""
生成csv文件
"""

import torch
import numpy as np
from manifm.model_pl import ManifoldFMLitModule
from omegaconf import OmegaConf

cfg_path = "outputs/runs/embed128/fm/2025.07.30/154602/.hydra/config.yaml"
ckpt_path = "outputs/runs/embed128/fm/2025.07.30/154602/checkpoints/epoch-070_step-0_loss--1.457715.ckpt"

cfg = OmegaConf.load(cfg_path)
model = ManifoldFMLitModule.load_from_checkpoint(ckpt_path, cfg=cfg)
model.eval().cuda()

n_samples = 50000
samples = model.sample(n_samples, device="cuda")
samples = samples.cpu().numpy()
samples_128 = samples[..., :128]

np.savetxt("generated_s128_03.csv", samples_128, delimiter=",")
print(f"✅ 已生成 {n_samples} 条 S¹²⁸ 数据：generated_s128.csv")