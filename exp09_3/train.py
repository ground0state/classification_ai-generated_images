# %%
import os
import random
from os.path import join

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config as c
from datasets import CustomDataset
from models import model_factory
from PIL import Image
from sklearn.model_selection import KFold, train_test_split
from tensorboardX import SummaryWriter  # TensorBoardのためのライブラリ
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import save_checkpoint, set_seed, setup_logger

# %%

# シードを設定
set_seed(c.seed)

# アノテーション読み込み
annotation_df = pd.read_csv(join(c.data_dir, c.annotation_filename))

# データの分割
train_df = annotation_df[annotation_df["group"] != c.val_group]
val_df = annotation_df[annotation_df["group"] == c.val_group]

# 保存用ディレクトリ作成
checkpoint_dir = f"../work_dirs/{c.exp}"
os.makedirs(checkpoint_dir, exist_ok=True)
log_dir = f"../work_dirs/{c.exp}/tf_logs"
os.makedirs(log_dir, exist_ok=False)  # 実験ミスを防ぐためにexist_ok=False

logger = setup_logger(join(checkpoint_dir, "history.log"))

train_dataset = CustomDataset(train_df, c.data_dir, is_train=True)
val_dataset = CustomDataset(val_df, c.data_dir, is_train=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=c.train_batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2)
val_loader = DataLoader(
    val_dataset,
    batch_size=c.val_batch_size,
    shuffle=False,
    num_workers=2)

# モデル、損失関数、最適化関数の設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model_factory()
model = model.to(device)

criterion = nn.CrossEntropyLoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=c.learning_rate)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=1.0)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=c.num_epochs*len(train_loader), eta_min=c.eta_min)

# %%
# 訓練ループ
# print("Training start.")
logger.info("Training start.")
# TensorBoardのためのライターをセットアップ
writer = SummaryWriter(logdir=log_dir)
for epoch in range(1, c.num_epochs+1):
    # Train loop
    model.train()
    total_loss = 0.0
    num_steps = len(train_loader)
    total_data = 0
    for step, (inputs, labels) in enumerate(train_loader, start=1):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_data += inputs.size(0)
        running_loss = total_loss / total_data

        if step % 50 == 0:
            # print(
            #     f"Epoch [{epoch}/{c.num_epochs}], Step [{step}/{num_steps}], Running Loss: {running_loss:.4f}")
            logger.info(f"Epoch [{epoch}/{c.num_epochs}], Step [{step}/{num_steps}], Running Loss: {running_loss:.4f}")
            # TensorBoardに訓練損失を記録
            writer.add_scalar("Train/Loss", running_loss,
                              (epoch-1) * num_steps + step)
            lr_ = scheduler.get_last_lr()[0]
            writer.add_scalar("Train/Lr", lr_,
                              (epoch-1) * num_steps + step)

        # 学習率スケジューラーのステップ
        scheduler.step()

    avg_train_loss = total_loss / total_data

    if (epoch <= 250 and epoch % 10 == 0) or epoch > 250:  # Hack: temporalなコード
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

        # print(
        #     f"Epoch [{epoch}/{c.num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        logger.info(f"Epoch [{epoch}/{c.num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        # TensorBoardに検証損失を記録
        writer.add_scalar("Val/Loss", avg_val_loss, epoch)
        # TensorBoardにエポックごとの検証精度を記録
        writer.add_scalar("Val/Accuracy", val_accuracy, epoch)

    if epoch >= 250:
        # チェックポイントの保存
        checkpoint_path = join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        save_checkpoint(epoch, model, optimizer, scheduler, path=checkpoint_path)



# print("Training complete.")
logger.info("Training complete.")

# %%
