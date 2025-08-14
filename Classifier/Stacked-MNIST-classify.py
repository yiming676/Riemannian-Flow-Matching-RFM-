import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import Counter
from itertools import product

class GrayMNISTClassifier(nn.Module):
    def __init__(self):
        super(GrayMNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.reshape(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GrayMNISTClassifier().to(device)
model.load_state_dict(torch.load("checkpoints/gray_mnist_cnn.pth", map_location=device))
model.eval()

npy_path = r"D:\riemannian-fm\outputs\mapper\recon\reconstructed.npy"
rgb_images = np.load(npy_path)  # (n, 3, 28, 28)
n = rgb_images.shape[0]
if rgb_images.shape[-1] == 3:
    rgb_images = np.transpose(rgb_images, (0, 3, 1, 2))  # (n, 3, 28, 28)


r = rgb_images[:, 0:1, :, :]  # (n, 1, 28, 28)
g = rgb_images[:, 1:2, :, :]
b = rgb_images[:, 2:3, :, :]

r_tensor = torch.tensor(r, dtype=torch.float32).to(device)
g_tensor = torch.tensor(g, dtype=torch.float32).to(device)
b_tensor = torch.tensor(b, dtype=torch.float32).to(device)

def predict(tensor):
    with torch.no_grad():
        logits = model(tensor)
        return torch.argmax(logits, dim=1).cpu().numpy()

r_digits = predict(r_tensor)
g_digits = predict(g_tensor)
b_digits = predict(b_tensor)

triplets = list(zip(r_digits, g_digits, b_digits))

df = pd.DataFrame(triplets, columns=["Red_digit", "Green_digit", "Blue_digit"])
df.to_csv("rgbmnist_digits.csv", index=False)
print("已保存每张图的三个数字到 rgbmnist_digits.csv")

all_digits = np.concatenate([r_digits, g_digits, b_digits])
counts = Counter(all_digits)

unique_triplets = set(triplets)
num_unique = len(unique_triplets)

total_possible = 10 * 10 * 10

coverage_rate = num_unique / total_possible

covered_triplets = set(triplets)
print(f"已覆盖组合数: {len(covered_triplets)} / 1000")

all_triplets = set(product(range(10), repeat=3))

missing_triplets = all_triplets - covered_triplets
print(f"未覆盖组合数: {len(missing_triplets)}")

covered_df = pd.DataFrame(sorted(covered_triplets), columns=["Red", "Green", "Blue"])
missing_df = pd.DataFrame(sorted(missing_triplets), columns=["Red", "Green", "Blue"])

covered_df.to_csv("covered_triplets.csv", index=False)
missing_df.to_csv("missing_triplets.csv", index=False)

print("已保存已覆盖组合到 covered_triplets.csv")
print("已保存未覆盖组合到 missing_triplets.csv")

print(f"RGB 组合覆盖率: {num_unique} / 1000 = {coverage_rate*100:.2f}%")

count_df = pd.DataFrame(sorted(counts.items()), columns=["Digit", "Count"])
count_df.to_csv("rgbmnist_digit_counts.csv", index=False)
#print("\n统计总数字类别如下：\n", count_df)
labels_array = np.stack([r_digits, g_digits, b_digits], axis=1)  # shape: (n, 3)

# 保存为 npy 文件
np.save("rgbmnist_labels.npy", labels_array)
print("已保存每张图的RGB通道标签到 rgbmnist_labels.npy")