import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


images_num = len(os.listdir('DIP_LicencePlate_MiniProject_Images'))
detected = 0
for img_path in os.listdir('DIP_LicencePlate_MiniProject_Images'):
    img = cv2.imread(os.path.join('DIP_LicencePlate_MiniProject_Images', img_path))
    width = int(img.shape[1] * 4)
    height = int(img.shape[0] * 4)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    enhanced = clahe.apply(gray)

    filtered = cv2.bilateralFilter(enhanced, 5, 50, 50)

    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY_INV, 15, 2)
        
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    opening[0:10, :] = 0 
    opening[-10:, :] = 0
    opening[:, 0:10] = 0
    opening[:, -10:] = 0 

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=4)

    output = img.copy()
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        if int(img.shape[0] * 0.15) < h < int(img.shape[0] * 0.7) and int(img.shape[1] * 0.035) < w < int(img.shape[1] * 0.2) and x > img.shape[1] * 0.07: 
            if int(img.shape[0] * 0.12 * img.shape[1] * 0.12) < w * h:
                if 0.2 < (w/h) < 2:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    detected = detected + 1

    os.makedirs('output', exist_ok=True)
    plt.imsave(os.path.join('output', img_path), cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

print((detected / (images_num * 8)) * 100)
