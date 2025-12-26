import cv2
import matplotlib.pyplot as plt

def histogram_equalization(image):
    h, w = image.shape
    total = h * w

    hist = [0] * 256
    for row in image:
        for px in row:
            hist[px] += 1

    pdf = [count / total for count in hist]

    cdf = [0.0] * 256
    running = 0.0
    for i in range(256):
        running += pdf[i]
        cdf[i] = running

    cdf_min = next((v for v in cdf if v>0), 0.0)
    lut = [0] * 256
    for i in range(256):
        val = (cdf[i] - cdf_min) / (1.0 - cdf_min) if (1.0 - cdf_min) != 0 else 0.0
        mapped = int(round(val * 255))
        if mapped < 0:
            mapped = 0
        elif mapped > 255:
            mapped = 255
        lut[i] = mapped

    out = [[0]*w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            out[i][j] = lut[image[i][j]]

    return out

img = cv2.imread('ai.jpg', cv2.IMREAD_GRAYSCALE)
equalized_img = histogram_equalization(img)

cv_equalized_img = cv2.equalizeHist(img)

plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("main image")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("equalized image")
plt.imshow(equalized_img, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("OpenCV equalized image")
plt.imshow(cv_equalized_img, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
