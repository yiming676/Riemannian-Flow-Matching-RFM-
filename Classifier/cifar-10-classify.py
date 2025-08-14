import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

image_folder = r'D:\riemannian-fm\cifar-10\cifar10_by_class\plane'
model_path = r'D:\riemannian-fm\cifar-10\saved_models\final_model.pth'
output_csv = './predictions.csv'

classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])


class Conv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        kwargs = {'kernel_size': 3, 'padding': 'same', 'bias': False, **kwargs}
        super().__init__(*args, **kwargs)


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12, momentum=0.4, weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class Linear(nn.Linear):
    def __init__(self, *args, temperature=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def forward(self, x):
        if self.temperature is not None:
            weight = self.weight * self.temperature
        else:
            weight = self.weight
        return x @ weight.T


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool1 = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x


class FastGlobalMaxPooling(nn.Module):
    def forward(self, x):
        return torch.amax(x, dim=(2, 3))


class SpeedyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        whiten_conv_depth = 24

        self.net_dict = nn.ModuleDict({
            'initial_block': nn.ModuleDict({
                'whiten': Conv(3, whiten_conv_depth, kernel_size=2, padding=0),
                'activation': nn.GELU(),
            }),
            'conv_group_1': ConvGroup(whiten_conv_depth, 64),  # 这里改为 24，不是48
            'conv_group_2': ConvGroup(64, 256),
            'conv_group_3': ConvGroup(256, 512),
            'pooling': FastGlobalMaxPooling(),
            'linear': Linear(512, 10, bias=False, temperature=1. / 9),
        })

    def forward(self, x):
        x = self.net_dict['initial_block']['whiten'](x)
        x = self.net_dict['initial_block']['activation'](x)
        x = self.net_dict['conv_group_1'](x)
        x = self.net_dict['conv_group_2'](x)
        x = self.net_dict['conv_group_3'](x)
        x = self.net_dict['pooling'](x)
        x = self.net_dict['linear'](x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpeedyConvNet().to(device)

checkpoint = torch.load(model_path, map_location=device)


if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
    state_dict = checkpoint['ema_state_dict']
else:
    state_dict = checkpoint

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v  # 去掉'module.'前缀
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model.eval()



def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        return classes[predicted.item()]

results = []
for filename in tqdm(os.listdir(image_folder)):
    if filename.lower().endswith('.png'):
        filepath = os.path.join(image_folder, filename)
        try:
            pred_class = predict_image(filepath)
            results.append({'filename': filename, 'predicted_class': pred_class})
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}")

correct = 0
total = 0
error_samples = []

for filename in tqdm(os.listdir(image_folder)):
    if filename.lower().endswith('.png'):
        filepath = os.path.join(image_folder, filename)
        try:
            pred_class = predict_image(filepath)
            true_class = os.path.basename(image_folder)  # 获取文件夹名作为真实类别
            results.append({'filename': filename, 'predicted_class': pred_class, 'true_class': true_class})

            # 统计准确率
            total += 1
            if pred_class == true_class:
                correct += 1
            else:
                error_samples.append(filename)  # 记录错误样本
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

accuracy = correct / total if total > 0 else 0
print(f"\n测试结果:")
print(f"样本总数: {total}")
print(f"正确预测: {correct}")
print(f"准确率: {accuracy:.2%}")

if error_samples:
    print("\n错误预测的样本:")
    for sample in error_samples:
        print(f"- {sample}")
else:
    print("\n所有样本预测正确")

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"\n预测结果已保存到 {output_csv}")