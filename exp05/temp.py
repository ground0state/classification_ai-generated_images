#%%
from torchvision import datasets, models, transforms

model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
# %%
model.head.in_features
# %%
