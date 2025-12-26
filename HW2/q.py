import numpy as np
import cv2
import os
import pandas as pd

def salt_noise_filter(shape, p=0.05):
    F = np.full(shape, -1, dtype=np.int16)
    mask = np.random.rand(*shape) < p
    F[mask] = 255
    return F

def pepper_noise_filter(shape, p=0.05):
    F = np.full(shape, -1, dtype=np.int16)
    mask = np.random.rand(*shape) < p
    F[mask] = 0
    return F

def salt_pepper_noise_filter(shape, p=0.05):
    F = np.full(shape, -1, dtype=np.int16)
    rand = np.random.rand(*shape)
    F[rand < p/2] = 0
    F[(rand >= p/2) & (rand < p)] = 255
    return F

def apply_noise_filter(f, F):
    g = f.copy()
    mask = F != -1
    g[mask] = F[mask]
    return g

def gaussian_noise_filter(shape, mean=0, sigma=20):
    F = np.random.normal(mean, sigma, shape)
    return F

def apply_gaussian_noise(f, F):
    g = f + F
    g = np.clip(g, 0, 255)
    return g.astype(np.uint8)

def uniform_noise_filter(shape, a=-30, b=30):
    F = np.random.uniform(a, b, shape)
    return F

def apply_uniform_noise(f, F):
    g = f + F
    g = np.clip(g, 0, 255)
    return g.astype(np.uint8)

def mean_filter(f, ksize=3):
    pad = ksize // 2
    h, w = f.shape
    f_padded = np.pad(f, pad, mode='edge')
    k = np.zeros_like(f, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            window = f_padded[i:i+ksize, j:j+ksize]
            k[i,j] = np.mean(window)

    return k.astype(np.uint8)

def median_filter(f, ksize=3):
    pad = ksize // 2
    h, w = f.shape
    f_padded = np.pad(f, pad, mode='edge')
    k = np.zeros_like(f)

    for i in range(h):
        for j in range(w):
            window = f_padded[i:i+ksize, j:j+ksize]
            k[i,j] = np.median(window)

    return k

def alpha_trimmed_mean_filter(f, ksize=3, alpha=2):
    pad = ksize // 2
    h, w = f.shape
    f_padded = np.pad(f, pad, mode='edge')
    k = np.zeros_like(f, dtype=np.float32)
    n = ksize*ksize

    for i in range(h):
        for j in range(w):
            window = f_padded[i:i+ksize, j:j+ksize].flatten()
            window.sort()
            trimmed = window[alpha//2 : n - alpha//2]
            k[i,j] = np.mean(trimmed)

    return k.astype(np.uint8)

def contraharmonic_mean_filter(f, ksize=3, Q=1.5):
    pad = ksize // 2
    h, w = f.shape
    f_padded = np.pad(f, pad, mode='edge')
    k = np.zeros_like(f, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            window = f_padded[i:i+ksize, j:j+ksize].astype(np.float32)
            num = np.sum(window ** (Q + 1))
            den = np.sum(window ** Q)
            k[i,j] = num / den if den != 0 else 0

    return np.clip(k, 0, 255).astype(np.uint8)

def count_fixed_pixels(f, g, k):
    noisy_mask = (g != f)

    fixed_mask = (k == f)

    return np.sum(noisy_mask & fixed_mask)

def count_new_errors(f, g, k):
    clean_mask = (g == f)

    damaged_mask = (k != f)

    return np.sum(clean_mask & damaged_mask)

data = []

for image_file in os.listdir('input'):
    f = cv2.imread(os.path.join('input', image_file), cv2.IMREAD_GRAYSCALE)

    name = os.path.splitext(image_file)[0]
    output_path = os.path.join('output', name)
    os.makedirs(output_path, exist_ok=True)

    F_salt = salt_noise_filter(f.shape, 0.05)
    F_pepper = pepper_noise_filter(f.shape, 0.05)
    F_sp = salt_pepper_noise_filter(f.shape, 0.1)
    F_gauss = gaussian_noise_filter(f.shape, sigma=15)
    F_uniform = uniform_noise_filter(f.shape, -20, 20)

    g_salt = apply_noise_filter(f, F_salt)
    g_pepper = apply_noise_filter(f, F_pepper)
    g_sp = apply_noise_filter(f, F_sp)
    g_gauss = apply_gaussian_noise(f, F_gauss)
    g_uniform = apply_uniform_noise(f, F_uniform)

    cv2.imwrite(os.path.join(output_path, "salt_noise.png"), g_salt)
    cv2.imwrite(os.path.join(output_path, "pepper_noise.png"), g_pepper)
    cv2.imwrite(os.path.join(output_path, "salt_pepper_noise.png"), g_sp)
    cv2.imwrite(os.path.join(output_path, "gaussian_noise.png"), g_gauss)
    cv2.imwrite(os.path.join(output_path, "uniform_noise.png"), g_uniform)

    filters = {
        "median": median_filter,
        "mean": mean_filter,
        "alpha": lambda x: alpha_trimmed_mean_filter(x, 3, 2),
        "contra": lambda x: contraharmonic_mean_filter(x, 3, 1.5),
    }

    noisy_images = {
        "salt": g_salt,
        "pepper": g_pepper,
        "sp": g_sp,
        "gauss": g_gauss,
        "uniform": g_uniform,
    }

    for noise_name,  g in noisy_images.items():
        for filter_name, filter_fn in filters.items():
            k = filter_fn(g)
            out_name = f"{noise_name}_{filter_name}.png"
            cv2.imwrite(os.path.join(output_path, out_name), k)
            fixed = count_fixed_pixels(f, g, k)
            errors = count_new_errors(f, g, k)
            data.append({
                "image": name,
                "noise": noise_name,
                "filter": filter_name,
                "fixed_pixels": fixed,
                "new_errors": errors
            })

df = pd.DataFrame(data)
df.to_csv("image_filter_results.csv", index=False)
