import tensorflow as tf
from model.utils import Con_Bn_Act
from tensorflow.keras import Model
from tensorflow.keras.layers import LeakyReLU, concatenate, Conv2D, Conv2DTranspose, BatchNormalization, Activation


class Generator(Model):
    def __init__(self, filters=64, layer_nums=8, output_channel=3):
        super(Generator, self).__init__()
        self.output_channel = output_channel
        self.layer_nums = layer_nums
        self.filters = filters

        self.con_act = LeakyReLU()

        self.down1_1 = Con_Bn_Act(filters=self.filters, activation=self.con_act)  # 512
        self.down1_2 = Down_Sample(filters=self.filters * 2)  # 256
        self.down2_1 = Con_Bn_Act(filters=self.filters, activation=self.con_act)  # 256
        self.down2_2 = Down_Sample(filters=self.filters * 2)  # 128
        self.down3_1 = Con_Bn_Act(filters=self.filters, activation=self.con_act)  # 128
        self.down3_2 = Down_Sample(filters=self.filters * 2)  # 64
        self.down4_1 = Con_Bn_Act(filters=self.filters, activation=self.con_act)  # 64
        self.down4_2 = Down_Sample(filters=self.filters * 2)  # 32
        self.down5_1 = Con_Bn_Act(filters=self.filters, activation=self.con_act)  # 32
        self.down5_2 = Down_Sample(filters=self.filters * 2)  # 16
        self.down6_1 = Con_Bn_Act(filters=self.filters, activation=self.con_act)  # 16
        self.down6_2 = Down_Sample(filters=self.filters * 2)  # 8
        self.down7_1 = Con_Bn_Act(filters=self.filters, activation=self.con_act)  # 8
        self.down7_2 = Down_Sample(filters=self.filters * 2)  # 4
        self.down8_1 = Con_Bn_Act(filters=self.filters, activation=self.con_act)  # 4
        self.down8_2 = Con_Bn_Act(filters=self.filters, activation=self.con_act)  # 4

        self.up8_3 = Con_Bn_Act(filters=self.filters, activation=self.con_act)  # 4
        self.up8_4 = Up_Sample(filters=self.filters)  # 8
        self.up7_3 = Con_Bn_Act(filters=self.filters, activation=self.con_act)  # 8   merge7_1
        self.up7_4 = Up_Sample(filters=self.filters)  # 16
        self.up6_3 = Con_Bn_Act(filters=self.filters, activation=self.con_act)  # 16  merge6_1
        self.up6_4 = Up_Sample(filters=self.filters)  # 32
        self.up5_3 = Con_Bn_Act(filters=self.filters, activation=self.con_act)  # 32  merge5_1
        self.up5_4 = Up_Sample(filters=self.filters)  # 64
        self.up4_3 = Con_Bn_Act(filters=self.filters, activation=self.con_act)  # 64  merge4_1
        self.up4_4 = Up_Sample(filters=self.filters)  # 128
        self.up3_3 = Con_Bn_Act(filters=self.filters, activation=self.con_act)  # 128 merge3_1
        self.up3_4 = Up_Sample(filters=self.filters)  # 256
        self.up2_3 = Con_Bn_Act(filters=self.filters, activation=self.con_act)  # 256 merge2_1
        self.up2_4 = Up_Sample(filters=self.filters)  # 512
        self.up1_3 = Con_Bn_Act(filters=self.filters, activation=self.con_act)  # 512 merge1_1
        self.up1_4 = Con_Bn_Act(filters=self.filters, activation=self.con_act)

        self.out = Conv2D(filters=self.output_channel,
                          kernel_size=3,
                          strides=1,
                          padding='same',
                          activation='tanh')  # 512

    def call(self, inputs, training=None, mask=None):
        down1_1 = self.down1_1(inputs)
        down1_2 = self.down1_2(down1_1)
        down2_1 = self.down2_1(down1_2)
        down2_2 = self.down2_2(down2_1)
        down3_1 = self.down3_1(down2_2)
        down3_2 = self.down3_2(down3_1)
        down4_1 = self.down4_1(down3_2)
        down4_2 = self.down4_2(down4_1)
        down5_1 = self.down5_1(down4_2)
        down5_2 = self.down5_2(down5_1)
        down6_1 = self.down6_1(down5_2)
        down6_2 = self.down6_2(down6_1)
        down7_1 = self.down7_1(down6_2)
        down7_2 = self.down7_2(down7_1)
        down8_1 = self.down8_1(down7_2)
        down8_2 = self.down8_2(down8_1)

        up8_3 = self.up8_3(down8_2)
        up8_4 = self.up8_4(up8_3)
        up7_3 = self.up7_3(concatenate([up8_4, down7_1], axis=3))
        up7_4 = self.up7_4(up7_3)
        up6_3 = self.up6_3(concatenate([up7_4, down6_1], axis=3))
        up6_4 = self.up6_4(up6_3)
        up5_3 = self.up5_3(concatenate([up6_4, down5_1], axis=3))
        up5_4 = self.up5_4(up5_3)
        up4_3 = self.up4_3(concatenate([up5_4, down4_1], axis=3))
        up4_4 = self.up4_4(up4_3)
        up3_3 = self.up3_3(concatenate([up4_4, down3_1], axis=3))
        up3_4 = self.up3_4(up3_3)
        up2_3 = self.up2_3(concatenate([up3_4, down2_1], axis=3))
        up2_4 = self.up2_4(up2_3)
        up1_3 = self.up1_3(concatenate([up2_4, down1_1], axis=3))
        up1_4 = self.up1_4(up1_3)

        out = self.out(up1_4)

        return out

# class Generator(Model):
#     def __init__(self, filters=64, layer_nums=8, output_channel=3):
#         super(Generator, self).__init__()
#         self.output_channel = output_channel
#         self.layer_nums = layer_nums
#         self.filters = filters
#
#         self.con_act = LeakyReLU()
#
#         self.down1 = Down_Sample(filters=64)  # 256
#         self.down2 = Down_Sample(filters=128)  # 128
#         self.down3 = Down_Sample(filters=256)  # 64
#         self.down4 = Down_Sample(filters=512)  # 32
#         self.down5 = Down_Sample(filters=512)  # 16
#         self.down6 = Down_Sample(filters=512)  # 8
#         self.down7 = Down_Sample(filters=512)  # 4
#         self.down8 = Down_Sample(filters=512)  # 2
#
#         self.up8 = Up_Sample(filters=512)  # 4 merge d8
#         self.up7 = Up_Sample(filters=512)  # 8 merge d7
#         self.up6 = Up_Sample(filters=512)  # 16 merge d6
#         self.up5 = Up_Sample(filters=512)  # 32 merge d5
#         self.up4 = Up_Sample(filters=512)  # 64 merge d4
#         self.up3 = Up_Sample(filters=256)  # 128 merge d3
#         self.up2 = Up_Sample(filters=128)  # 256 merge d2
#         self.up1 = Up_Sample(filters=64)  # 512 merge d1
#
#         self.out = Conv2D(filters=self.output_channel,
#                           kernel_size=3,
#                           strides=1,
#                           padding='same',
#                           activation='tanh')  # 512
#
#     def call(self, inputs, training=None, mask=None):
#         down1 = self.down1(inputs)
#         down2 = self.down2(down1)
#         down3 = self.down3(down2)
#         down4 = self.down4(down3)
#         down5 = self.down5(down4)
#         down6 = self.down6(down5)
#         down7 = self.down7(down6)
#         down8 = self.down8(down7)
#
#         up8 = self.up8(down8)
#         up7 = self.up7(concatenate([up8, down7], axis=3))
#         up6 = self.up6(concatenate([up7, down6], axis=3))
#         up5 = self.up5(concatenate([up6, down5], axis=3))
#         up4 = self.up4(concatenate([up5, down4], axis=3))
#         up3 = self.up3(concatenate([up4, down3], axis=3))
#         up2 = self.up2(concatenate([up3, down2], axis=3))
#         up1 = self.up1(concatenate([up2, down1], axis=3))
#
#         out = self.out(up1)
#
#         return out


class Vgg19_Discriminator(Model):
    def __init__(self, input_shape, requires_grad=False, mixed_precision=False):
        super(Vgg19_Discriminator, self).__init__()
        m_tf = tf.keras.applications.vgg19.VGG19(include_top=False, input_shape=input_shape)
        layer_ids = [2, 5, 8, 13, 18]
        if mixed_precision:
            base_model_outputs = [Activation('linear', dtype='float32')(m_tf.layers[id].output) for id in
                                  layer_ids]
        else:
            base_model_outputs = [m_tf.layers[id].output for id in layer_ids]
        self.model = Model(inputs=m_tf.input, outputs=base_model_outputs)
        self.model.trainable = False

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)


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


class Down_Sample(Model):
    def __init__(self, filters):
        super().__init__()
        self.down = Con_Bn_Act(filters=filters,
                               strides=2,
                               kernel_initializer=tf.random_normal_initializer(0., 0.02),
                               activation=LeakyReLU())

    def call(self, inputs, training=None, mask=None):
        out = self.down(inputs)

        return out


class Up_Sample(Model):
    def __init__(self, filters):
        super().__init__()
        self.con_transpose = Conv2DTranspose(filters=filters,
                                             kernel_size=3,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                             use_bias=False)
        self.bn = BatchNormalization()
        self.act = LeakyReLU()

    def call(self, inputs, training=None, mask=None):
        con_transpose = self.con_transpose(inputs)
        bn = self.bn(con_transpose)
        out = self.act(bn)

        return out