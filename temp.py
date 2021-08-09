import os
from glob import glob

img_file_list = glob('./data/food_datasets/food_023/*.jpg')
img_file_list = img_file_list[:3]

print(len(img_file_list))
for index, img_file in enumerate(img_file_list):
    os.rename(img_file, './data/food_datasets/food_023/' + str(index) + '.jpg')
