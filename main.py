import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
import matplotlib.pyplot as plt
from matplotlib import cm
import os

def show_hough_transform(image, filename):
    low_dark = (0, 0, 0)
    high_dark = (50, 50, 50)
    only_dark = cv2.inRange(img, low_dark, high_dark)
    h, theta, d = hough_line(canny(only_dark)) # вычисляем преобразование Хаффа от границ изображения

    fig, ax = plt.subplots(1, 3, figsize=(15, 6))

    ax[0].imshow(only_dark, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(np.log(1 + h),
                 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                 cmap='gray', aspect=1/20)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')

    ax[2].imshow(image, cmap=cm.gray)
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
        ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
    ax[2].set_xlim((0, image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def width_door_hough(image):
    low_dark = (0, 0, 0)
    high_dark = (50, 50, 50)
    Eps = 0.1
    only_dark = cv2.inRange(img, low_dark, high_dark)
    dists = []
    h, theta, d = hough_line(canny(only_dark)) # вычисляем преобразование Хаффа от границ изображения
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        if angle  > -Eps and angle < Eps:
            dists.append(dist)
    if len(dists) >= 2:
        return max(dists) - min(dists)
    else:
        return 0


directory = os.fsencode("Datas3")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"): 
        img = cv2.imread("Datas3\\" + filename)
        #img = rgb2gray(cv2.imread("Datas\\" + filename))
        img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
        show_hough_transform(img, filename)
        #print('Door wight: ', width_door_hough(img))


cv2.waitKey(0)
cv2.destroyAllWindows()


