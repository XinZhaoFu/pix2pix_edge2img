from model.discriminator import Discriminator
from model.generator import Generator
import numpy as np
import tensorflow as tf
import cv2


class Pix2pix_predicter:
    def __init__(self, ex_name, checkpoint_dir):
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

    def predict(self):
        self.checkpoint.restore(self.ck_manager.latest_checkpoint)
        if self.ck_manager.latest_checkpoint:
            print("[info]Restored from {}".format(self.ck_manager.latest_checkpoint))
        else:
            print("[info]Initializing from scratch.")

        test_input = np.zeros(shape=(1, 256, 256, 3), dtype=np.float32)
        ori_img = cv2.imread('./data/val/img/apple_pie_003158095.jpg')
        ori_img = cv2.resize(ori_img, dsize=(256, 256))
        img = (ori_img / 127.5) - 1
        test_input[0:, :, :, :] = img[:, :, :]
        prediction = self.generator(test_input, training=False)

        prediction = (prediction * 0.5 + 0.5) * 256
        prediction_output = np.empty(shape=(256, 256, 3), dtype=np.uint8)
        prediction_output[:, :, :] = prediction[0:, :, :, :]
        # print(prediction_output)
        res = np.zeros(shape=(256, 512, 3), dtype=np.uint8)
        res[:, :256, :] = ori_img
        res[:, 256:, :] = prediction_output

        cv2.imwrite('./data/res/test1.jpg', res)


def main():
    checkpoint_dir = './checkpoints/pix2pix_checkpoints/'
    ex_name = 'pix2pix_256'
    predicter = Pix2pix_predicter(ex_name, checkpoint_dir)
    predicter.predict()


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    main()
