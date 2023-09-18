#%%
import pandas as pd
import cv2
from matplotlib import pyplot as plt

im = cv2.imread(f"/media/data/gen_orig_clas/train_1/train_1.png")
# %%
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# %%
# edges = cv2.Canny(im, 100, 200)
edges = cv2.Sobel(im,         # 入力画像
                            cv2.CV_32F,  # ビット深度
                            1,           # x方向に微分
                            1,           # y方向に微分
                            ksize=3      # カーネルサイズ(3 x 3)
                        )
# %%
plt.imshow(edges, 'gray')

# %%
from torchvision import models
import torch.nn as nn

model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
# %%
model
# %%
