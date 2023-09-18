import os
from os.path import join

import numpy as np
import pandas as pd
import torchvision.utils as vutils
from config import Config as c
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# データロードおよび前処理
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.ColorJitter(brightness=0.1, contrast=0,
    #                        saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class CustomDataset(Dataset):
    def __init__(
        self,
        annotation_df,
        img_dir,
        is_train
    ):
        self.records = annotation_df.to_dict(orient="records")
        self.img_dir = img_dir

        if is_train:
            self.transform = train_transforms
            self.target_transform = None
        else:
            self.transform = val_transforms
            self.target_transform = None

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


def create_tiled_image(dataset, n, save_path):
    """
    Create a tiled image from N samples of the dataset.
    """

    # DataLoaderを使用してデータセットからN枚の画像を取得
    dataloader = DataLoader(dataset, batch_size=n, shuffle=True)
    images, _ = next(iter(dataloader))

    # N枚の画像をタイル状に並べる
    tiled_image = vutils.make_grid(images, nrow=int(np.sqrt(n)))

    # [-1, 1] の範囲から [0, 1] の範囲に変更
    tiled_image = (tiled_image - tiled_image.min()) / \
        (tiled_image.max() - tiled_image.min())

    # 保存
    vutils.save_image(tiled_image, save_path)


if __name__ == "__main__":
    annotation_df = pd.read_csv(join(c.data_dir, c.annotation_filename))
    dataset = CustomDataset(annotation_df, c.data_dir, is_train=True)
    os.makedirs(f"../work_dirs/{c.exp}", exist_ok=True)
    create_tiled_image(dataset, 12, f"../work_dirs/{c.exp}/train_image_samples.png")
