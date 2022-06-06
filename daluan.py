from PIL import Image
import numpy as np

import cv2
import numpy as np
import matplotlib.pyplot as plt


im = Image.open("/share/home/wangzy/PycharmProjects/PyTorch-Deep-Image-Steganography/PyTorch-Deep-Image-Steganography/orisecret.png")
RGB=np.array(im)

# 返回矩阵的行和列
s_1 = RGB.shape[0]
r = np.random.choice(s_1, size=s_1, replace=False, p=None)
RGBS = RGB[r, :, :]
s_2 = RGB.shape[1]
c = np.random.choice(s_2, size=s_2, replace=False, p=None)
RGBSS = RGBS[:, c, :]


#jiemi
i = 0
f = np.arange(0, len(c))
while i <= len(c):
    # find（r==j）是返回r中等于j的值的索引，可以是多个，赋值给f的第j个位置
    for (k, val) in enumerate(c):
        if val == i:
            f[i] = k
    i = i + 1

j = 0
g = np.arange(0, len(r))
RGBE = RGBSS[:, f, :]
while j <= len(r):
    # find（r==j）是返回r中等于j的值的索引，可以是多个，赋值给f的第j个位置
    for (k, val) in enumerate(r):
        if val == j:
            g[j] = k
    j = j + 1

RGBEE = RGBE[g, :, :]




plt.subplot(1, 3, 1)
plt.imshow(RGB)
plt.subplot(1, 3, 2)
plt.imshow(RGBSS)
plt.title(u"加密后")
plt.subplot(1, 3, 3)
plt.imshow(RGBEE)
plt.title(u"解密后")
plt.show()
