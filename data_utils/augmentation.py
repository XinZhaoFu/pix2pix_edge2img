import cv2
from glob import glob
from tqdm import tqdm
from data_utils.utils import shuffle_file, mosaic, random_crop


def augmentation():
    aug_img_save_path = '../data/food_datasets/img_aug/'
    aug_label_save_path = '../data/food_datasets/label_aug/'

    img_file_list = glob('../data/food_datasets/img/*.*')
    label_file_list = glob('../data/food_datasets/label/*.*')

    assert len(img_file_list) == len(label_file_list)

    img_file_list.sort()
    label_file_list.sort()
    img_file_list, label_file_list = shuffle_file(img_file_list, label_file_list)

    for img_file, label_file in tqdm(zip(img_file_list, label_file_list), total=len(img_file_list)):
        img = cv2.imread(img_file)
        label = cv2.imread(label_file)
        img_name = (img_file.split('/')[-1]).split('.')[0]

        # ori
        cv2.imwrite(aug_img_save_path + img_name + '.jpg', img)
        cv2.imwrite(aug_label_save_path + img_name + '.jpg', label)

        # random_crop
        crop_img, crop_label = random_crop(img, label)
        cv2.imwrite(aug_img_save_path + img_name + '_crop.jpg', crop_img)
        cv2.imwrite(aug_label_save_path + img_name + '_crop.jpg', crop_label)

    for _ in range(4):
        temp_img_file_list, temp_label_file_list = shuffle_file(img_file_list, label_file_list)
        img_file_list.extend(temp_img_file_list)
        label_file_list.extend(temp_label_file_list)

    mosaic_img_list, mosaic_label_list, mosaic_name_list = [], [], []
    for img_file, label_file in tqdm(zip(img_file_list, label_file_list), total=len(img_file_list)):
        img = cv2.imread(img_file)
        label = cv2.imread(label_file)
        img_name = (img_file.split('/')[-1]).split('.')[0]

        # mosaic
        mosaic_img_list.append(img)
        mosaic_label_list.append(label)
        mosaic_name_list.append(img_name)

        if len(mosaic_img_list) == 4:
            mosaic_img, mosaic_label = mosaic(ori_images=mosaic_img_list, ori_labels=mosaic_label_list, res_size=512)
            mosaic_name = mosaic_name_list[0][:5] + mosaic_name_list[1][:5] + \
                          mosaic_name_list[2][:5] + mosaic_name_list[3][:5]
            mosaic_img_list, mosaic_label_list, mosaic_name_list = [], [], []
            cv2.imwrite(aug_img_save_path + mosaic_name + '.jpg', mosaic_img)
            cv2.imwrite(aug_label_save_path + mosaic_name + '.jpg', mosaic_label)


if __name__ == '__main__':
    augmentation()
