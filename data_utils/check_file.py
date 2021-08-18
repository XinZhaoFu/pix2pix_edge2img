import cv2
from glob import glob
from tqdm import tqdm
import numpy as np
import os

img_file_list = glob('../data/food_datasets/img/*.*')
label_file_list = glob('../data/food_datasets/label/*.*')

print(len(img_file_list), len(label_file_list))

img_file_list.sort()
label_file_list.sort()

for img_file, label_file in zip(img_file_list, label_file_list):
    img = cv2.imread(img_file)
    label = cv2.imread(label_file)
    img_rows, img_cols, _ = img.shape

    if img.shape != label.shape:
        # print('------------')
        print(img_file, label_file)
        os.remove(img_file)
        os.remove(label_file)
        # print('-------------')
        # continue

    img_np = np.array(img)
    label_np = np.array(label)

    dis_np = img_np - label_np
    dis_sum = np.sum(dis_np)
    if int(dis_sum / (img_rows * img_cols)) > 400:
        print(str(int(dis_sum / (img_rows * img_cols))) + '\t' + img_file + '\t' + label_file)
        img_name = img_file.split('/')[-1]

        res_np = np.zeros(shape=(img_rows, img_cols * 2, 3), dtype=np.uint8)
        res_np[:, :img_cols] = img
        res_np[:, img_cols:] = label

        cv2.imwrite('../data/check/' + img_name, res_np)
