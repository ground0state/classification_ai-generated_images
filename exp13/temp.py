#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 画像を読み込む
filepath = "/media/data/gen_orig_clas/train_1/train_16.png"
img = cv2.imread(filepath)

# カラー画像をRGBに変換
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# パワースペクトルを計算する関数
def compute_power_spectrum(channel):
    # 2D フーリエ変換
    f = np.fft.fft2(channel)
    # ゼロ周波数成分を画像の中心に移動
    fshift = np.fft.fftshift(f)
    # パワースペクトルを計算
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum

# RGBチャネルごとのパワースペクトルを計算
channels = cv2.split(img_rgb)
magnitude_spectrums = [compute_power_spectrum(channel) for channel in channels]

# 各チャネルのパワースペクトルを結合してRGB画像を作成
magnitude_img = cv2.merge(magnitude_spectrums)
magnitude_img = magnitude_img.astype(np.uint8)

# 結果を表示
plt.imshow(magnitude_img)
plt.title('Power Spectrum')
plt.show()

# %%
magnitude_img
# %%
