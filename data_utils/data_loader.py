import tensorflow as tf
from glob import glob
from data_utils.utils import shuffle_file


class Data_Loader:
    def __init__(self,
                 buffer_size=420,
                 batch_size=1,
                 size=256):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.size = size
        self.train_img_path = './data/train/img/*.jpg'
        self.train_label_path = './data/train/label/*.jpg'
        self.val_img_path = './data/val/img/*.jpg'
        self.val_label_path = './data/val/label/*.jpg'

        self.train_img_file_list = glob(self.train_img_path)
        self.train_label_file_list = glob(self.train_label_path)
        self.val_img_file_list = glob(self.val_img_path)
        self.val_label_file_list = glob(self.val_label_path)

        assert len(self.train_label_file_list) > 0 and len(self.train_img_file_list) > 0 \
               and len(self.val_img_file_list) > 0 and len(self.val_label_file_list) > 0
        assert len(self.train_label_file_list) == len(self.train_img_file_list)
        assert len(self.val_img_file_list) == len(self.val_label_file_list)

        print('[info] train_img_num:\t' + str(len(self.train_img_file_list))
              + '\ttrain_label_num:\t' + str(len(self.train_label_file_list))
              + '\tval_img_num:\t' + str(len(self.val_img_file_list))
              + '\tval_label_num:\t' + str(len(self.val_label_file_list)))

    def load_image_label(self, img_file, label_file):
        img = tf.io.read_file(img_file)
        img = tf.image.decode_jpeg(img)
        label = tf.io.read_file(label_file)
        label = tf.image.decode_jpeg(label)

        img = tf.cast(img, tf.float32)
        label = tf.cast(label, tf.float32)

        img = tf.image.resize(img, [self.size, self.size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        label = tf.image.resize(label, [self.size, self.size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img = (img / 127.5) - 1
        label = (label / 127.5) - 1

        return img, label

    def get_dataset(self, img_file_list, label_file_list):
        img_file_list.sort()
        label_file_list.sort()
        img_file_list, label_file_list = shuffle_file(img_file_list, label_file_list)

        dataset = tf.data.Dataset.from_tensor_slices((img_file_list, label_file_list))
        dataset = dataset.map(self.load_image_label, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    def get_train_datasets(self):
        train_datasets = self.get_dataset(self.train_img_file_list, self.train_label_file_list)

        return train_datasets

    def get_val_datasets(self):
        val_datasets = self.get_dataset(self.val_img_file_list, self.val_label_file_list)

        return val_datasets
