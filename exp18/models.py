# %%
from torchvision import models
import torch.nn as nn


# def model_factory():
#     model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1)
#     num_ftrs = model.head.in_features
#     model.head = nn.Linear(num_ftrs, 2)
#     return model


# def model_factory():
#     model = models.efficientnet_v2_s(
#         weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
#     num_ftrs = model.classifier[1].in_features
#     model.classifier[1] = nn.Linear(num_ftrs, 2)
#     return model

def model_factory():
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model
