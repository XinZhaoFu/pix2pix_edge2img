from glob import glob
from shutil import copyfile
from tqdm import tqdm
from utils import shuffle_file, recreate_dir


def distribution_datasets():
    split_rate = 0.98

    datasets_img_path = '../data/food_datasets/img_aug/'
    datasets_label_path = '../data/food_datasets/label_aug/'

    train_img_path = '../data/train/img/'
    train_label_path = '../data/train/label/'
    val_img_path = '../data/val/img/'
    val_label_path = '../data/val/label/'
    recreate_dir(train_img_path)
    recreate_dir(train_label_path)
    recreate_dir(val_img_path)
    recreate_dir(val_label_path)

    ds_img_file_list = glob(datasets_img_path + '*.*')
    ds_label_file_list = glob(datasets_label_path + '*.*')

    assert len(ds_img_file_list) == len(ds_label_file_list)
    ds_img_file_list.sort()
    ds_label_file_list.sort()

    ds_img_file_list, ds_label_file_list = shuffle_file(ds_img_file_list, ds_label_file_list)

    train_img_file_list = ds_img_file_list[:int(len(ds_img_file_list) * split_rate)]
    train_label_file_list = ds_label_file_list[:int(len(ds_img_file_list) * split_rate)]
    val_img_file_list = ds_img_file_list[int(len(ds_img_file_list) * split_rate):]
    val_label_file_list = ds_label_file_list[int(len(ds_img_file_list) * split_rate):]

    for ds_img_file, ds_label_file in tqdm(zip(train_img_file_list, train_label_file_list),
                                           total=len(train_label_file_list)):
        ds_img_name = ds_img_file.split('/')[-1]
        ds_label_name = ds_label_file.split('/')[-1]

        copyfile(ds_img_file, train_img_path + ds_img_name)
        copyfile(ds_label_file, train_label_path + ds_label_name)

    for ds_img_file, ds_label_file in tqdm(zip(val_img_file_list, val_label_file_list),
                                           total=len(val_label_file_list)):
        ds_img_name = ds_img_file.split('/')[-1]
        ds_label_name = ds_label_file.split('/')[-1]

        copyfile(ds_img_file, val_img_path + ds_img_name)
        copyfile(ds_label_file, val_label_path + ds_label_name)


if __name__ == '__main__':
    distribution_datasets()
