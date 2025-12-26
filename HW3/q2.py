import cv2
import numpy as np
import matplotlib.pyplot as plt

def DFT1D(signal):
    N = len(signal)
    X = np.zeros(N, dtype=complex)
    for u in range(N):
        s = 0
        for x in range(N):
            angle = -2j * np.pi * ((u * x) / N)
            s += signal[x] * np.exp(angle)
        X[u] = s
    return X


def IDFT1D(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        s = 0
        for k in range(N):
            angle = 2j * np.pi * (k * n) / N
            s += X[k] * np.exp(angle)
        x[n] = s / N
    return x


def DFT2D(img):
    M, N = img.shape

    row_trans = np.zeros((M, N), dtype=complex)
    for x in range(M):
        row_trans[x, :] = DFT1D(img[x, :])

    F = np.zeros((M, N), dtype=complex)
    for y in range(N):
        F[:, y] = DFT1D(row_trans[:, y])
    return F



def IDFT2D(F):
    M, N = F.shape
    temp = np.zeros((M, N), dtype=complex)

    for j in range(N):
        temp[:, j] = IDFT1D(F[:, j])

    img = np.zeros((M, N), dtype=complex)
    for i in range(M):
        img[i, :] = IDFT1D(temp[i, :])

    return img.real


def shift_image(img):
    M, N = img.shape
    out = np.zeros_like(img, dtype=float)
    for x in range(M):
        for y in range(N):
            out[x, y] = img[x, y] * ((-1) ** (x + y))
    return out


def generate_periodic_noise(M, N, A=50, k1=15, k2=15):
    noise = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            noise[u, v] = A * np.cos(2 * np.pi * (k1 * u / M + k2 * v / N))
    return noise

wave_img = cv2.imread("3.jpg", cv2.IMREAD_GRAYSCALE).astype(float)
target = cv2.imread("4.bmp", cv2.IMREAD_GRAYSCALE).astype(float)
wave_img = cv2.resize(wave_img, (target.shape[1], target.shape[0]))

F_wave = DFT2D(shift_image(wave_img))

rows, cols = wave_img.shape
cx, cy = rows // 2, cols // 2

y = np.arange(rows)
x = np.arange(cols)
X, Y = np.meshgrid(x, y)
R = np.sqrt((X - cy)**2 + (Y - cx)**2)

r1 = 15
r2 = 80

mask = np.logical_and(R > r1, R < r2).astype(float)
F_extracted = F_wave * mask

wave_extracted_spatial = shift_image(IDFT2D(wave_img))

F = DFT2D(shift_image(wave_img))
F_out = F + F_extracted

out_img = shift_image(IDFT2D(F_out))
out_img = np.clip(out_img, 0, 255).astype(np.uint8)

plt.figure(figsize=(12,6))

plt.subplot(2,3,1)
plt.imshow(wave_img, cmap='gray')
plt.title("Wave Input")

plt.subplot(2,3,2)
plt.imshow(mask, cmap='gray')
plt.title("Radial Mask")

plt.subplot(2,3,3)
plt.imshow(wave_extracted_spatial, cmap='gray')
plt.title("Extracted Wave (Spatial)")

plt.subplot(2,3,4)
plt.imshow(target, cmap='gray')
plt.title("Target Image")

plt.subplot(2,3,5)
plt.imshow(np.log(1 + np.abs(F_out)), cmap='gray')
plt.title("Frequency Domain (Combined)")

plt.subplot(2,3,6)
plt.imshow(out_img, cmap='gray')
plt.title("Final Result (Waves Added)")

plt.show()
