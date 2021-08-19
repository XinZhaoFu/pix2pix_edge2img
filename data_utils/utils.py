from random import randint

import cv2
import numpy as np
import os
import shutil


def shuffle_file(img_file_list, label_file_list):
    """
    打乱img和label的文件列表顺序 并返回两列表 seed已固定

    :param img_file_list:
    :param label_file_list:
    :return:
    """
    np.random.seed(10)
    index = [i for i in range(len(img_file_list))]
    np.random.shuffle(index)
    img_file_list = np.array(img_file_list)[index]
    label_file_list = np.array(label_file_list)[index]
    img_file_list = img_file_list.tolist()
    label_file_list = label_file_list.tolist()
    return img_file_list, label_file_list


def create_dir(folder_name):
    """
    创建文件夹

    :param folder_name:
    :return:
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print('[INFO] 新建文件夹：' + folder_name)


def recreate_dir(folder_name):
    """
    重建文件夹

    :param folder_name:
    :return:
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        shutil.rmtree(folder_name)
        os.makedirs(folder_name)
    print('[INFO] 重建文件夹：' + folder_name)


def random_crop(ori_img, ori_label):
    """
    随机裁剪

    :param ori_img:
    :param ori_label:
    :return:
    """
    if ori_img.shape != ori_label.shape:
        return ori_img, ori_label

    crop_img = np.array(ori_img, dtype=np.uint8)
    crop_label = np.array(ori_label, dtype=np.uint8)
    img_rows, img_cols, _ = ori_img.shape
    rows_random_init = randint(int(img_rows * 0.05), int(img_rows * 0.2))
    rows_random_end = randint(int(img_rows * 0.8), int(img_rows * 0.95))
    cols_random_init = randint(int(img_cols * 0.05), int(img_cols * 0.2))
    cols_random_end = randint(int(img_cols * 0.8), int(img_cols * 0.95))
    crop_img = crop_img[rows_random_init:rows_random_end, cols_random_init:cols_random_end, :]
    crop_label = crop_label[rows_random_init:rows_random_end, cols_random_init:cols_random_end, :]

    return crop_img, crop_label


def random_crop_assign_size(ori_img, ori_label, crop_img_rows, crop_img_cols):
    """
    随机裁剪 指定尺寸

    :param ori_img:
    :param ori_label:
    :param crop_img_rows:
    :param crop_img_cols:
    :return:
    """
    if ori_img.shape != ori_label.shape:
        return ori_img, ori_label

    img_rows, img_cols, _ = ori_img.shape
    if img_rows < crop_img_rows or img_cols < crop_img_cols:
        ori_img = cv2.resize(ori_img, dsize=(crop_img_cols, crop_img_rows))
        ori_label = cv2.resize(ori_label, dsize=(crop_img_cols, crop_img_rows))
        return ori_img, ori_label

    crop_img = np.array(ori_img, dtype=np.uint8)
    crop_label = np.array(ori_label, dtype=np.uint8)

    crop_img_rows = min(img_rows, crop_img_rows)
    crop_img_cols = min(img_cols, crop_img_cols)

    rows_random_init = randint(0, img_rows - crop_img_rows)
    rows_random_end = rows_random_init + crop_img_rows
    cols_random_init = randint(0, img_cols - crop_img_cols)
    cols_random_end = cols_random_init + crop_img_cols

    crop_img = crop_img[rows_random_init:rows_random_end, cols_random_init:cols_random_end, :]
    crop_label = crop_label[rows_random_init:rows_random_end, cols_random_init:cols_random_end, :]

    return crop_img, crop_label


def mosaic(ori_images, ori_labels, res_size=512):
    """
    多图拼接 原文中是对一个batch中的图片进行拼接 这里是离线的数据增强

    :param ori_images:
    :param ori_labels:
    :param res_size:
    :return:
    """
    ori_img1, ori_img2, ori_img3, ori_img4 = ori_images
    ori_label1, ori_label2, ori_label3, ori_label4 = ori_labels

    res_img = np.zeros(shape=(res_size, res_size, 3), dtype=np.uint8)
    res_label = np.zeros(shape=(res_size, res_size, 3), dtype=np.uint8)

    center_random_row = randint(int(res_size * 0.2), int(res_size * 0.8))
    center_random_col = randint(int(res_size * 0.2), int(res_size * 0.8))

    crop_img1, crop_label1 = random_crop_assign_size(ori_img1, ori_label1, center_random_row, center_random_col)
    crop_img2, crop_label2 = random_crop_assign_size(ori_img2, ori_label2,
                                                     res_size - center_random_row, center_random_col)
    crop_img3, crop_label3 = random_crop_assign_size(ori_img3, ori_label3,
                                                     center_random_row, res_size - center_random_col)
    crop_img4, crop_label4 = random_crop_assign_size(ori_img4, ori_label4,
                                                     res_size - center_random_row, res_size - center_random_col)

    res_img[:center_random_row, :center_random_col, :] = crop_img1
    res_label[:center_random_row, :center_random_col, :] = crop_label1
    res_img[center_random_row:, :center_random_col, :] = crop_img2
    res_label[center_random_row:, :center_random_col, :] = crop_label2
    res_img[:center_random_row, center_random_col:, :] = crop_img3
    res_label[:center_random_row, center_random_col:, :] = crop_label3
    res_img[center_random_row:, center_random_col:, :] = crop_img4
    res_label[center_random_row:, center_random_col:, :] = crop_label4

    return res_img, res_label


def random_flip(img, label):
    """
    做一个随机翻转

    :param img:
    :param label:
    :return:
    """
    random_num = randint(0, 1)
    img = cv2.flip(img, random_num)
    label = cv2.flip(label, random_num)

    return img, label


def random_rotate(img, label):
    """
    做一个随机旋转 但只有0 90 180 和270

    :param img:
    :param label:
    :return:
    """
    img_rows, img_cols, _ = img.shape
    label_rows, label_cols, _ = label.shape
    random_num = randint(0, 3)
    img_rotate = cv2.getRotationMatrix2D((img_rows * 0.5, img_cols * 0.5), 90 * random_num, 1)
    label_rotate = cv2.getRotationMatrix2D((label_rows * 0.5, label_cols * 0.5), 90 * random_num, 1)
    if random_num == 2:
        img = cv2.warpAffine(img, img_rotate, (img_rows, img_cols))
        label = cv2.warpAffine(label, label_rotate, (label_rows, label_cols))
    else:
        img = cv2.warpAffine(img, img_rotate, (img_cols, img_rows))
        label = cv2.warpAffine(label, label_rotate, (label_cols, label_rows))

    return img, label

