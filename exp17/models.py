from torchvision import models
import torch.nn as nn


def model_factory():
    model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, 2)
    return model
