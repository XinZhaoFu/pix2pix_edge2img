import os
import numpy as np
from glob import glob
from random import randint

# img_file_list = glob('./data/food_datasets/food_023/*.jpg')
# img_file_list = img_file_list[:3]
#
# print(len(img_file_list))
# for index, img_file in enumerate(img_file_list):
#     os.rename(img_file, './data/food_datasets/food_023/' + str(index) + '.jpg')

# img_name = 'frozen_yogurt_000738108.jpg'
# os.remove('./data/food_datasets/img/' + img_name)
# os.remove('./data/food_datasets/label/' + img_name)

# np_temp = np.random.rand(10, 10)
# np_temp = np_temp[2:8, 3:7]
# print(np_temp)

for _ in range(20):
    random_num = randint(0, 3)
    print(random_num)
