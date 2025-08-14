import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm

data_dir = r"D:\\riemannian-fm\\tiny-imagenet-200"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    weights.transforms()
])
val_transforms = weights.transforms()

train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transforms)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4)

model = convnext_tiny(weights=weights)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 200)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

def train(model, train_loader, val_loader, epochs=50):
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"[Train Epoch {epoch+1}/{epochs}]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=total_loss/total, acc=correct/total)

        acc = validate(model, val_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "convnext_tiny_tiny_imagenet.pth")
            print(f"[Saved best model with acc: {best_acc:.2f}%]")

def validate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    acc = 100. * correct / total
    print(f"[Validation Accuracy] {acc:.2f}%")
    return acc

if __name__ == '__main__':
    train(model, train_loader, val_loader, epochs=50)