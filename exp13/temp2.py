#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2

def compute_power_spectrum(img):
    # 2D フーリエ変換
    f = np.fft.fft2(img)
    # ゼロ周波数成分を画像の中心に移動
    fshift = np.fft.fftshift(f)
    # パワースペクトルを計算
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum

# 画像を読み込む
filepath = "/media/data/gen_orig_clas/train_1/train_1.png"

img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

# パワースペクトルを計算
magnitude_spectrum = compute_power_spectrum(img)

# 結果を表示
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Power Spectrum')
plt.colorbar()
plt.show()

# %%
