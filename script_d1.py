import os

import numpy as np
import skimage.io as io
from skimage.transform import rotate
from skimage.util import random_noise


def get_dataset(path, augmentation=False):
    X = []
    y = []
    for filename in os.listdir(path):
        image = io.imread(path + filename)
        image = image / 255
        label = '_'.join(filename.split('_')[1:4])
        X.append(image)
        print(image)
        if augmentation:
            X.append(rotate(image, angle=45, mode='wrap'))
            X.append(np.fliplr(image))
            X.append(np.flipud(image))
            X.append(random_noise(image, var=0.2 ** 2))
        for j in range(5 if augmentation else 1):
            y.append(label)
        break
    return np.array(X), np.array(y)


path = 'faces_2/'
X, y = get_dataset(path)
print(y)
