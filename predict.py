from model.discriminator import Discriminator
from model.generator import Generator
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
from glob import glob


class Pix2pix_predicter:
    def __init__(self, ex_name, checkpoint_dir, data_size):
        self.data_size = data_size
        self.ex_name = ex_name
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.checkpoint_dir = checkpoint_dir

        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        self.ck_manager = tf.train.CheckpointManager(self.checkpoint,
                                                     directory=self.checkpoint_dir,
                                                     max_to_keep=1,
                                                     checkpoint_name=self.ex_name + '_ck')

        self.checkpoint.restore(self.ck_manager.latest_checkpoint)
        if self.ck_manager.latest_checkpoint:
            print("[info]Restored from {}".format(self.ck_manager.latest_checkpoint))
        else:
            print("[info]Initializing from scratch.")

    def predict(self, img):
        test_input = np.zeros(shape=(1, self.data_size, self.data_size, 3), dtype=np.float32)
        img = (img / 127.5) - 1
        test_input[0:, :, :, :] = img[:, :, :]

        prediction = self.generator(test_input, training=False)

        prediction = (prediction * 0.5 + 0.5) * 256
        prediction_output = np.empty(shape=(self.data_size, self.data_size, 3), dtype=np.uint8)
        prediction_output[:, :, :] = prediction[0:, :, :, :]

        return prediction_output


def main():
    ex_name = 'pix2pix_512'
    checkpoint_dir = './checkpoints/pix2pix512_checkpoints/'
    data_size = 512

    val_img_path = './data/val/img/'
    val_label_path = './data/val/label/'
    save_path = './data/res/'

    val_img_file_list = glob(val_img_path + '*.*')
    val_label_file_list = glob(val_label_path + '*.*')
    assert len(val_img_file_list) == len(val_label_file_list)
    val_img_file_list.sort()
    val_label_file_list.sort()

    predicter = Pix2pix_predicter(ex_name=ex_name, checkpoint_dir=checkpoint_dir, data_size=data_size)

    for val_img_file, val_label_file in tqdm(zip(val_img_file_list, val_label_file_list), total=len(val_img_file_list)):
        val_img = cv2.imread(val_img_file)
        val_label = cv2.imread(val_label_file)
        val_img = cv2.resize(val_img, dsize=(data_size, data_size))
        val_label = cv2.resize(val_label, dsize=(data_size, data_size))
        res_name = val_img_file.split('/')[-1]
        predict_img = predicter.predict(val_img)

        res = np.zeros(shape=(data_size, data_size * 3, 3), dtype=np.uint8)
        res[:, :data_size, :] = val_img
        res[:, data_size:data_size*2] = val_label
        res[:, data_size*2:, :] = predict_img

        cv2.imwrite(save_path + res_name, res)


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    main()
