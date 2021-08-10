from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, ReLU, Conv2DTranspose, LeakyReLU, concatenate
import tensorflow as tf
from model.utils import Con_Bn_Act
from model.pix2pix import Down_Sample


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
            block = ResnetBlock(filters=self.filters * 4)
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

#
# class Discriminator(Model):
#     def __init__(self, input_shape):
#         super(Discriminator, self).__init__()
#         self.vgg19 = tf.keras.applications.vgg19.VGG19(include_top=False, input_shape=input_shape)
#         # layer_ids = [2, 5, 8, 13, 18]
#
#         # base_model_outputs = [m_tf.layers[layer_id].output for layer_id in layer_ids]
#         # self.model = Model(inputs=m_tf.input, outputs=base_model_outputs)
#         # self.model.trainable = False
#
#     def call(self, inputs, training=None, mask=None):
#         # return self.model(inputs)
#
#         out = self.vgg19(inputs)


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.down1 = Down_Sample(filters=64)
        self.down2 = Down_Sample(filters=128)
        self.down3 = Down_Sample(filters=256)
        # self.down4 = Down_Sample(filters=256)

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
