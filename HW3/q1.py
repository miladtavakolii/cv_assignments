import numpy as np
import cv2
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


img = cv2.imread("1.bmp", 0).astype(float)


f = shift_image(img)

F = DFT2D(f)

M, N = img.shape
noise = generate_periodic_noise(M, N, A=80, k1=20, k2=20)

f_noise = shift_image(noise)
F_noise = DFT2D(f_noise)

F_noisy = F + F_noise

f1 = IDFT2D(F_noisy)

g = shift_image(f1)

plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Frequency Spectrum")
plt.imshow(np.log(1 + np.abs(F)), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Noise Frequency Spectrum")
plt.imshow(np.log(np.abs(F_noise)), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Final Output")
plt.imshow(g, cmap='gray')
plt.axis('off')

plt.show()
