import datetime
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from model.generator import Generator
from data_utils.data_loader import get_dataset


class Pix2pix_predicter:
    def __init__(self):
        self.test_dataset = get_dataset()
        self.generator = Generator()

        self.checkpoint_dir = './checkpoints/pix2pix_checkpoints.ckpt'
        self.checkpoint = tf.train.Checkpoint()
        self.log_dir = './log/'
        self.summary_writer = tf.summary.create_file_writer(
            self.log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    def predict(self):
        for inp, tar in self.test_dataset.take(5):
            self.generate_images(self.generator, inp, tar)

    def generate_images(self,model, test_input, tar):
        prediction = model(test_input, training=False)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
            plt.savefig('image_at_epoch_{:04d}.png'.format(i))


def main():
    predicter = Pix2pix_predicter()
    predicter.predict()


if __name__ == '__main__':
    main()



