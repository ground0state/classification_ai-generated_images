import logging
import os
import random
from datetime import datetime

import numpy as np
import pytz
import torch


class JSTFormatter(logging.Formatter):
    """JST (日本標準時) でログの時刻を表示するカスタムフォーマッタ"""
    converter = datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        dt = dt.astimezone(pytz.timezone('Asia/Tokyo'))
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.strftime(self.default_time_format)
        return s


def setup_logger(save_path):
    # ロガーの設定
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # フォーマットの設定
        # JSTでのフォーマット設定
    formatter = JSTFormatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S %Z')

    # 標準出力用のハンドラの設定
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # ファイル出力用のハンドラの設定
    fh = logging.FileHandler(save_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(epoch, model, optimizer, scheduler, path="checkpoint.bin"):
    """モデル、オプティマイザ、スケジューラ、ランダム状態などを保存する関数"""

    # DataParallelを使用している場合はmodel.moduleを取り出す。
    model_to_save = model.module if hasattr(model, "module") else model

    checkpoint = {
        "epoch": epoch,
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(),  # numpy.randomを使用する場合は必要
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        "cuda_random": torch.cuda.get_rng_state(),  # gpuを使用する場合は必要
        "cuda_random_all": torch.cuda.get_rng_state_all(),  # 複数gpuを使用する場合は必要
    }

    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, scheduler, path="checkpoint.bin"):
    """保存されたチェックポイントをロードし、モデル、オプティマイザ、スケジューラ、ランダム状態などを復元する関数"""

    checkpoint = torch.load(path)

    if hasattr(model, "module"):  # DataParallelを使用した場合
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])
    epoch = checkpoint["epoch"]
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    random.setstate(checkpoint["random"])
    np.random.set_state(checkpoint["np_random"])
    torch.set_rng_state(checkpoint["torch"])
    torch.random.set_rng_state(checkpoint["torch_random"])
    torch.cuda.set_rng_state(checkpoint["cuda_random"])  # gpuを使用する場合は必要
    torch.cuda.set_rng_state_all(
        checkpoint["cuda_random_all"])  # 複数gpuを使用する場合は必要

    return epoch, model, optimizer, scheduler
