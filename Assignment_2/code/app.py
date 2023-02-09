import os
import cv2
import numpy as np


def image_stats(image):
    l, a, b = cv2.split(image)
    return (l.mean(), l.std(), a.mean(), a.std(), b.mean(), b.std())


def color_transfer(source, target):
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype('float32')
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype('float32')

    (lMeansrc, lStdsrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    (l, a, b) = cv2.split(target)

    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    l = (lStdTar * l / lStdsrc)
    a = (aStdTar * a / aStdSrc)
    b = (bStdTar * b / bStdSrc)

    l += lMeansrc
    a += aMeanSrc
    b += bMeanSrc

    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    transfer = cv2.merge([l, a, b])

    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    return transfer


def get_files():
    source_images = []
    target_images = []
    result_images = []
    for filename in os.listdir("data"):
        if filename.endswith(".png") and filename.startswith('in'):
            source = cv2.imread('data/' + filename)
            source_images.append(source)

        elif filename.endswith(".png") and filename.startswith('tar'):
            target = cv2.imread('data/' + filename)

            target_images.append(target)

        elif filename.endswith(".png") and filename.startswith('res'):
            result = cv2.imread('data/' + filename)
            result_images.append(result)
    return source_images, target_images, result_images


sources, targets, results =  get_files()

for i in range(len(sources)):
    transfer = color_transfer(targets[i], sources[i])
    original_result = results[i]
    cv2.imshow('Trasfer', transfer)
    cv2.imwrite(f'./results/transfered_{i}.png', transfer)
    cv2.imwrite(f'./results/original_{i}.png', original_result)

    cv2.imshow('original_result ', original_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
