from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, ReLU, Conv2DTranspose
import tensorflow as tf
from model.utils import Con_Bn_Act


class Generator(Model):
    def __init__(self, filters=64, residual_num=7, output_channel=3):
        super(Generator, self).__init__()
        self.filters = filters
        self.residual_num = residual_num
        self.output_channel = output_channel
        self.kernel_initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = Con_Bn_Act(filters=self.filters, kernel_size=7, kernel_initializer=self.kernel_initializer)
        self.down2 = Con_Bn_Act(filters=self.filters * 2, kernel_initializer=self.kernel_initializer, strides=2)
        self.down3 = Con_Bn_Act(filters=self.filters * 4, kernel_initializer=self.kernel_initializer, strides=2)

        self.res_blocks = Sequential()
        for _ in range(self.residual_num):
            block = ResnetBlock(filters=self.filters)
            self.res_blocks.add(block)

        self.up3 = Up_Block(filters=self.filters * 4)
        self.up2 = Up_Block(filters=self.filters * 2)
        self.up1 = Conv2D(filters=self.output_channel,
                          kernel_size=7,
                          kernel_initializer=self.kernel_initializer,
                          padding='same',
                          activation='tanh')

    def call(self, inputs, training=None, mask=None):
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        res_block = self.res_blocks(down3)

        up3 = self.up3(res_block)
        up2 = self.up2(up3)
        out = self.up1(up2)

        return out


class ResnetBlock(Model):
    def __init__(self, filters):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.cbr1 = Con_Bn_Act(filters=self.filters, kernel_initializer=tf.random_normal_initializer(0., 0.02))
        self.cbr2 = Con_Bn_Act(filters=self.filters, kernel_initializer=tf.random_normal_initializer(0., 0.02),
                               activation='not')

    def call(self, inputs, training=None, mask=None):
        cbr1 = self.cbr1(inputs)
        cbr2 = self.cbr2(cbr1)

        return cbr2 + inputs


class Up_Block(Model):
    def __init__(self, filters):
        super(Up_Block, self).__init__()
        self.filters = filters
        self.con_transpose = Conv2DTranspose(filters=self.filters,
                                             kernel_size=3,
                                             padding='same',
                                             strides=2,
                                             kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                             use_bias=False)
        self.bn = BatchNormalization()
        self.act = ReLU()

    def call(self, inputs, training=None, mask=None):
        con_transpose = self.con_transpose(inputs)
        bn = self.bn(con_transpose)
        out = self.act(bn)

        return out
