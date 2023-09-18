import argparse
import os
from os.path import join

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from config import Config as c
from models import model_factory
from PIL import Image
from tqdm import tqdm


class Predictor:
    def __init__(self, checkpoint_path=None):
        self.model = model_factory()
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

        # モデルを評価モードに設定
        self.model.eval()

        # 学習済みの重みを読み込む
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])

        # 入力画像の前処理を定義
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

    def predict_proba(self, image_path):
        # 画像の読み込み
        image = Image.open(image_path)

        # 前処理
        image = self.transform(image).unsqueeze(0)  # バッチ次元を追加
        image = image.to(self.device)

        # 予測の実行
        with torch.no_grad():
            outputs = self.model(image)
            probs = F.softmax(outputs, dim=1)  # ソフトマックス関数で確率を計算
            # クラス1の確率を取得
            predicted_prob = probs[0][1].item()
        return predicted_prob


def save_predictions_to_csv(image_dir, images, checkpoint_path, save_file_path):
    predictor = Predictor(checkpoint_path)
    with open(save_file_path, 'w', encoding="utf-8") as f:
        for filename in tqdm(images):
            image_path = join(image_dir, filename)
            prob = predictor.predict_proba(image_path)
            f.write(f"{filename},{prob}\n")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parse command line arguments.")
    parser.add_argument('--image_sub_dir', type=str,
                        required=True, help='Sub directory for images')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for the save filename')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    image_sub_dir = args.image_sub_dir
    suffix = args.suffix

    image_dir = join(c.data_dir, image_sub_dir)
    images = os.listdir(image_dir)
    checkpoint_path = join(
        c.work_dirs, c.exp, f"checkpoint_epoch_{c.test_epoch}.pth")

    if suffix != "":
        save_filename = f"submission_{c.exp}_val_group_{c.val_group}.csv"
    else:
        save_filename = f"submission_{c.exp}_val_group_{c.val_group}_{suffix}.csv"
    save_file_path = join(c.work_dirs, c.exp, save_filename)

    print("Test start.")
    save_predictions_to_csv(image_dir, images, checkpoint_path, save_file_path)
    print("Test complete.")
