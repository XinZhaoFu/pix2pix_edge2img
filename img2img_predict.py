# coding=utf-8
import glob
import cv2
import os
from model.unet import UNet
import tensorflow as tf
import numpy as np
import datetime
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.set_printoptions(threshold=np.inf)

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)


def edge_predict():
    """

    :return:
    """
    print('[info]模型加载 图片加载')
    # 加载模型
    model = UNet(semantic_filters=64,
                 detail_filters=64,
                 output_channels=3,
                 semantic_num_cbr=1,
                 detail_num_cbr=4,
                 end_activation='tanh')
    checkpoint_save_path = './checkpoints/unet512_64/unet512.ckpt'
    model.load_weights(checkpoint_save_path)
    test_file_path_list = glob.glob('./data/train/img/*.jpg')
    label_file_path_list = glob.glob('./data/train/label/*.jpg')
    test_file_path_list.sort()
    label_file_path_list.sort()
    test_file_path_list = test_file_path_list[:1]
    label_file_path_list = label_file_path_list[:1]

    for test_file, label_file in tqdm(zip(test_file_path_list, label_file_path_list), total=len(test_file_path_list)):

        ori_test_img = cv2.imread(test_file)
        ori_test_img = cv2.resize(ori_test_img, dsize=(512, 512))
        label_img = cv2.imread(label_file)
        label_img = cv2.resize(label_img, dsize=(512, 512))

        test_img = ori_test_img / 255.
        test_img_temp = np.empty(shape=(1, 512, 512, 3))
        test_img_temp[0, :, :, :] = test_img[:, :, :]

        predict = model.predict(test_img_temp)

        prediction = predict * 255
        prediction_output = np.empty(shape=(512, 512, 3), dtype=np.uint8)
        prediction_output[:, :, :] = prediction[0:, :, :, :]

        res = np.zeros(shape=(512, 512 * 3, 3), dtype=np.uint8)
        res[:, :512, :] = ori_test_img
        res[:, 512:1024, :] = prediction_output
        res[:, 1024:, :] = label_img

        cv2.imwrite('./data/img2img_res/demo1.jpg', res)


def main():
    start_time = datetime.datetime.now()

    edge_predict()

    end_time = datetime.datetime.now()
    print('time:\t' + str(end_time - start_time).split('.')[0])


if __name__ == '__main__':
    main()
