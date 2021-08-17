import cv2
from glob import glob
from tqdm import tqdm
import numpy as np
from random import randint

img_file_list = glob('../data/food_datasets/img/*.*')
label_file_list = glob('../data/food_datasets/label/*.*')

assert len(img_file_list) == len(label_file_list)

for img_file, label_file in tqdm(zip(img_file_list, label_file_list), total=len(img_file_list)):
    img = cv2.imread(img_file)
    label = cv2.imread(label_file)
    img_name = (img_file.split('/')[-1]).split('.')[0]





def random_crop(ori_img, ori_label):
    if ori_img.shape != ori_label.shape:
        return ori_img, ori_label

    crop_img = np.array(ori_img, dtype=np.uint8)
    img_rows, img_cols, _ = ori_img.shape
    rows_random_init = randint(int(img_rows * 0.05), int(img_rows * 0.2))
    rows_random_end = randint(int(img_rows * 0.8), int(img_rows * 0.95))
    cols_random_init = randint(int(img_cols * 0.05), int(img_cols * 0.2))
    cols_random_end = randint(int(img_cols * 0.8), int(img_cols * 0.95))
    crop_img = crop_img[rows_random_init:rows_random_end, cols_random_init:cols_random_end]

    return crop_img


def random_crop_assign_size(ori_img, ori_label, crop_img_rows, crop_img_cols):
    if ori_img.shape != ori_label.shape:
        return ori_img, ori_label

    crop_img = np.array(ori_img, dtype=np.uint8)
    img_rows, img_cols, _ = ori_img.shape
    # crop_img_rows = min(img_rows, crop_img_rows)
    # crop_img_cols = min(img_cols, crop_img_cols)
    #
    # rows_random_init = randint(0, img_rows - crop_img_rows)
    # rows_random_end = rows_random_init + crop_img_rows
    # cols_random_init = randint(0, )
    # cols_random_end = randint(int(img_cols * 0.8), int(img_cols * 0.95))
