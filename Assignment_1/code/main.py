import numpy as np
from skimage import img_as_float
from skimage.io import imread, imsave, imshow
import os

def bgr_split(img_name):

    image = imread(img_name)
    image = img_as_float(image)
    height = np.floor(image.shape[0] / 3.0).astype(np.int)

    blue = image[:height]
    green = image[height: 2 * height]
    red = image[2 * height: 3 * height]

    return blue, green, red



def align_layers(image_1, image_2):

    image_1_crop = image_1[int(0.1 * len(image_1)):-int(0.1 * len(image_1)),
            int(0.1 * len(image_1[0])):-int(0.1 * len(image_1[0]))]
    image_2_crop = image_2[int(0.1 * len(image_2)):-int(0.1 * len(image_2)),
            int(0.1 * len(image_2[0])):-int(0.1 * len(image_2[0]))]

    best_score = -float('inf')
    best_shift = [0, 0]

    for i in range(-15, 16):
        for j in range(-15, 16):
            temp_score = score(np.roll(image_1_crop, (i, j), (0, 1)), image_2_crop)
            if temp_score > best_score:
                best_score = temp_score
                best_shift = [i, j]

    return np.roll(image_1, best_shift, (0, 1))



def score(image_1, image_2):
    return -np.sum(np.sum((image_1 - image_2) ** 2))



for filename in os.listdir("code/data"):
    if filename.endswith(".jpg"):
        blue, green, red = bgr_split("code/data/" + filename)

        aligned_green = align_layers(green, blue)
        aligned_red = align_layers(red, blue)

        image_out = np.dstack([aligned_red, aligned_green, blue])

        imsave("code/output/" + filename, image_out)
