import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import imageio
import cv2
import pickle
from random import shuffle

all_xray_df = pd.read_csv('C:\\Users\\hpc\\Desktop\\MMD\\ChexNet\\Dataset\\Data_Entry_2017.csv')
img_path = os.path.join('C:\\Users\\hpc\\Desktop\\MMD\\ChexNet\\Dataset\\images')

# img_fps = list(set(glob.glob(img_path+'/'+'*.png')))

os.chdir('..\\Dataset\\images')

Image_Index = list(set(glob.glob('*.png')))
shuffle(Image_Index)

os.chdir('..\\..\\Codes')

img_fps = []
for inx in Image_Index:
    img_fps.append(img_path + '\\' + str(inx))

width = 224
height = 224

resized_images = []
j = 0
resized_images.append(cv2.resize(imageio.imread(img_fps[0]), (width, height)))
for i in range(1, len(img_fps)):
    resized_images.append(cv2.resize(imageio.imread(img_fps[i]), (width, height)))
    if i % 10000 == 0:
        my_dict = {Image_Index[k]: resized_images[k % 10000] for k in range(i - 10000, i)}
        f = open('dict' + str(j) + '.p', 'wb')
        pickle.dump(my_dict, f)
        f.close()
        j = j + 1
        resized_images = []
        print(i)
