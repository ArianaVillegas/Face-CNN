import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import pandas as pd
import cv2
import os

path = '/home/bryan/Imágenes/faces/'
files = os.listdir(path)


output_path = '/home/bryan/Imágenes/output'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('Directorio creado:', output_path)



count = 0
for file in files:
    image_path = path + '/' + file
    image = cv2.imread(image_path)
    gaussian_noise=iaa.AdditiveGaussianNoise(150,180) 
    noise_image=gaussian_noise.augment_image(image)
    rotate=iaa.Affine(rotate=(-70, 100))
    rotated_image=rotate.augment_image(noise_image)
    gaussian_noise2=iaa.AdditiveGaussianNoise(250,300) 
    noise_image2=gaussian_noise2.augment_image(image)
    crop = iaa.Crop(percent=(0, 0.2)) # crop image
    corp_image=crop.augment_image(noise_image2) 
    flip_vr=iaa.Flipud(p=1.0)
    flip_vr_image= flip_vr.augment_image(noise_image)
    contrast=iaa.GammaContrast(gamma=6)
    contrast_image =contrast.augment_image(image)
    #ia.imshow(contrast_image)
    cv2.imwrite(output_path + '/faces' + '_' +  str(count) + '_' + file, noise_image)
    count += 1