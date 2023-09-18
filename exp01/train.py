# %%
import os
from os.path import join

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import KFold, train_test_split
from tensorboardX import SummaryWriter  # TensorBoardのためのライブラリ
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


def save_checkpoint(epoch, model, optimizer, path="checkpoint.pth"):
    """チェックポイントを保存する関数"""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(state, path)

class CustomDataset(Dataset):
    def __init__(
        self,
        annotation_df,
        img_dir,
        transform=None,
        target_transform=None
    ):
        self.records = annotation_df.to_dict(orient="records")
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        img_path = os.path.join(self.img_dir, record["path"])
        image = Image.open(img_path)

        label = record["category"]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# %%
DATA_DIR = "/media/data/gen_orig_clas"
annotation_df = pd.read_csv(join(DATA_DIR, "all.csv"))

val_group = 1
train_df = annotation_df[annotation_df["group"] != val_group]
val_df = annotation_df[annotation_df["group"] == val_group]
# %%


# Parameters -------------------------------
num_epochs = 10
train_batch_size = 64
val_batch_size = 64
learning_rate = 0.001 / train_batch_size
exp = "exp01"
# ------------------------------------------

checkpoint_dir = f"../work_dirs/{exp}"
os.makedirs(checkpoint_dir)
log_dir = f"../work_dirs/{exp}/tf_logs"
os.makedirs(log_dir)

# データロードおよび前処理
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(train_df, DATA_DIR, train_transforms)
val_dataset = CustomDataset(val_df, DATA_DIR, val_transforms)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=2)

# モデル、損失関数、最適化関数の設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# %%
# 訓練ループ
print("Training start.")
# TensorBoardのためのライターをセットアップ
writer = SummaryWriter(logdir=log_dir)
for epoch in range(1, num_epochs+1):
    # Train loop
    model.train()
    total_loss = 0.0
    num_steps = len(train_loader)
    for step, (inputs, labels) in enumerate(train_loader, start=1):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        running_loss = total_loss / step

        if step % 50 == 0:
            print(
                f"Step [{step}/{num_steps}], Running Loss: {running_loss:.4f}")
            # TensorBoardに訓練損失を記録
            writer.add_scalar("Train/Loss", running_loss,
                              (epoch-1) * num_steps + step)

    avg_train_loss = total_loss / len(train_loader.dataset)

    # Validation loop
    model.eval()
    total_val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    avg_val_loss = total_val_loss / len(val_loader.dataset)
    val_accuracy = 100 * correct / len(val_loader.dataset)

    print(
        f"Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    # TensorBoardに検証損失を記録
    writer.add_scalar("Val/Loss", avg_val_loss, epoch)
    # TensorBoardにエポックごとの検証精度を記録
    writer.add_scalar("Val/Accuracy", val_accuracy, epoch)

    # チェックポイントの保存
    checkpoint_path = join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    save_checkpoint(epoch, model, optimizer, path=checkpoint_path)

print("Training complete.")

# %%
