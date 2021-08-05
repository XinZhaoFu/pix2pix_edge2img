import tensorflow as tf
from model.utils import Down_Sample, Con_Bn_Act
from tensorflow.keras import Model
from tensorflow.keras.layers import LeakyReLU, concatenate, Conv2D


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.down1 = Down_Sample(filters=64)
        self.down2 = Down_Sample(filters=128)
        self.down3 = Down_Sample(filters=256)

        self.con = Con_Bn_Act(filters=512,
                              kernel_size=3,
                              strides=1,
                              kernel_initializer=tf.random_normal_initializer(0., 0.02),
                              activation=LeakyReLU(),
                              padding='same')

        self.out = Conv2D(filters=1,
                          kernel_size=3,
                          strides=1,
                          kernel_initializer=tf.random_normal_initializer(0., 0.02),
                          padding='same')

    def call(self, inputs, training=None, mask=None):
        [inp, tar] = inputs
        concat = concatenate([inp, tar], axis=3)
        down1 = self.down1(concat)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        con = self.con(down3)
        out = self.out(con)

        return out
