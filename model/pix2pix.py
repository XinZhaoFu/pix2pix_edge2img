import tensorflow as tf
from model.unet import Up_CBR_Block
from model.utils import Con_Bn_Act, CBR_Block
from tensorflow.keras import Model
from tensorflow.keras.layers import LeakyReLU, concatenate, Conv2D, Conv2DTranspose, BatchNormalization, ReLU, \
    MaxPooling2D, Activation


class MUnet_Generator(Model):
    def __init__(self,
                 semantic_filters=32,
                 detail_filters=64,
                 output_channels=3,
                 detail_num_cbr=4,
                 end_activation='sigmoid'):
        super(MUnet_Generator, self).__init__()
        self.semantic_filters = semantic_filters
        self.output_channels = output_channels
        self.detail_num_cbr = detail_num_cbr
        self.end_activation = end_activation
        self.detail_filters = detail_filters

        self.cbr_block1_1 = Con_Bn_Act(filters=self.detail_filters)
        self.cbr_block1_2 = Con_Bn_Act(filters=self.semantic_filters * 2, strides=2)
        self.cbr_block2_1 = Con_Bn_Act(filters=self.semantic_filters)
        self.cbr_block2_2 = Con_Bn_Act(filters=self.semantic_filters * 2, strides=2)
        self.cbr_block3_1 = Con_Bn_Act(filters=self.semantic_filters)
        self.cbr_block3_2 = Con_Bn_Act(filters=self.semantic_filters * 2, strides=2)
        self.cbr_block4_1 = Con_Bn_Act(filters=self.semantic_filters)
        self.cbr_block4_2 = Con_Bn_Act(filters=self.semantic_filters * 2, strides=2)
        self.cbr_block5_1 = Con_Bn_Act(filters=self.semantic_filters)
        self.cbr_block5_2 = Con_Bn_Act(filters=self.semantic_filters * 2, strides=2)
        self.cbr_block6_1 = Con_Bn_Act(filters=self.semantic_filters)
        self.cbr_block6_2 = Con_Bn_Act(filters=self.semantic_filters * 2, strides=2)
        self.cbr_block7 = Con_Bn_Act(filters=self.semantic_filters)

        self.cbr_block_up7 = Up_CBR_Block(filters=self.semantic_filters)
        self.cbr_block_up6 = Up_CBR_Block(filters=self.semantic_filters)
        self.cbr_block_up5 = Up_CBR_Block(filters=self.semantic_filters)
        self.cbr_block_up4 = Up_CBR_Block(filters=self.semantic_filters)
        self.cbr_block_up3 = Up_CBR_Block(filters=self.semantic_filters)
        self.cbr_block_up2 = Up_CBR_Block(filters=self.semantic_filters)
        self.cbr_block_up1 = Con_Bn_Act(filters=self.semantic_filters)

        self.cbr_block_detail = CBR_Block(filters=self.detail_filters, num_cbr=self.detail_num_cbr,
                                          block_name='detail')

        self.con_end = Con_Bn_Act(filters=self.output_channels, activation=self.end_activation)

    def call(self, inputs, training=None, mask=None):
        con1 = self.cbr_block1_1(inputs)
        detail = self.cbr_block_detail(con1)

        pool2 = self.cbr_block1_2(con1)
        con2 = self.cbr_block2_1(pool2)

        pool3 = self.cbr_block2_2(con2)
        con3 = self.cbr_block3_1(pool3)

        pool4 = self.cbr_block3_2(con3)
        con4 = self.cbr_block4_1(pool4)

        pool5 = self.cbr_block4_2(con4)
        con5 = self.cbr_block5_1(pool5)

        pool6 = self.cbr_block5_2(con5)
        con6 = self.cbr_block6_1(pool6)

        pool7 = self.cbr_block6_2(con6)
        con7 = self.cbr_block7(pool7)

        up7 = self.cbr_block_up7(con7)

        merge6 = concatenate([up7, con6], axis=3)
        up6 = self.cbr_block_up6(merge6)

        merge5 = concatenate([up6, con5], axis=3)
        up5 = self.cbr_block_up5(merge5)

        merge4 = concatenate([up5, con4], axis=3)
        up4 = self.cbr_block_up4(merge4)

        merge3 = concatenate([up4, con3], axis=3)
        up3 = self.cbr_block_up3(merge3)

        merge2 = concatenate([up3, con2], axis=3)
        up2 = self.cbr_block_up2(merge2)

        merge1 = concatenate([up2, detail], axis=3)
        up1 = self.cbr_block_up1(merge1)

        out = self.con_end(up1)

        return out


class Unet_Generator(Model):
    def __init__(self,
                 semantic_filters=64,
                 detail_filters=64,
                 output_channels=3,
                 semantic_num_cbr=1,
                 detail_num_cbr=4,
                 end_activation='sigmoid'):
        super(Unet_Generator, self).__init__()
        self.semantic_filters = semantic_filters
        self.output_channels = output_channels
        self.semantic_num_cbr = semantic_num_cbr
        self.detail_num_cbr = detail_num_cbr
        self.end_activation = end_activation
        self.detail_filters = detail_filters

        self.cbr_block1 = CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='down1')
        self.cbr_block2 = CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='down2')
        self.cbr_block3 = CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='down3')
        self.cbr_block4 = CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='down4')
        self.cbr_block5 = CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='down5')
        self.cbr_block6 = CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='down6')
        self.cbr_block7 = CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='down7')

        self.cbr_block_up7 = Up_CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr,
                                          block_name='up7')
        self.cbr_block_up6 = Up_CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr,
                                          block_name='up6')
        self.cbr_block_up5 = Up_CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr,
                                          block_name='up5')
        self.cbr_block_up4 = Up_CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr,
                                          block_name='up4')
        self.cbr_block_up3 = Up_CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr,
                                          block_name='up3')
        self.cbr_block_up2 = Up_CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr,
                                          block_name='up2')
        self.cbr_block_up1 = CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='up1')

        self.cbr_block_detail = CBR_Block(filters=self.detail_filters, num_cbr=self.detail_num_cbr, block_name='detail')

        self.con_end = Con_Bn_Act(filters=self.output_channels, activation=self.end_activation)

        self.pool = MaxPooling2D(padding='same')

    def call(self, inputs, training=None, mask=None):
        con1 = self.cbr_block1(inputs)
        detail = self.cbr_block_detail(con1)

        pool2 = self.pool(con1)
        con2 = self.cbr_block2(pool2)

        pool3 = self.pool(con2)
        con3 = self.cbr_block3(pool3)

        pool4 = self.pool(con3)
        con4 = self.cbr_block4(pool4)

        pool5 = self.pool(con4)
        con5 = self.cbr_block5(pool5)

        pool6 = self.pool(con5)
        con6 = self.cbr_block6(pool6)

        pool7 = self.pool(con6)
        con7 = self.cbr_block7(pool7)

        up7 = self.cbr_block_up7(con7)

        merge6 = concatenate([up7, con6], axis=3)
        up6 = self.cbr_block_up6(merge6)

        merge5 = concatenate([up6, con5], axis=3)
        up5 = self.cbr_block_up5(merge5)

        merge4 = concatenate([up5, con4], axis=3)
        up4 = self.cbr_block_up4(merge4)

        merge3 = concatenate([up4, con3], axis=3)
        up3 = self.cbr_block_up3(merge3)

        merge2 = concatenate([up3, con2], axis=3)
        up2 = self.cbr_block_up2(merge2)

        merge1 = concatenate([up2, detail], axis=3)
        up1 = self.cbr_block_up1(merge1)

        out = self.con_end(up1)

        return out


class Generator(Model):
    def __init__(self, filters=64, layer_nums=8, output_channel=3):
        super(Generator, self).__init__()
        self.output_channel = output_channel
        self.layer_nums = layer_nums
        self.filters = filters

        self.con_act = ReLU()

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
                          activation='sigmoid')  # 512

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
#                           kernel_initializer=tf.random_normal_initializer(0., 0.02),
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


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.down1 = Down_Sample(filters=64)
        self.down2 = Down_Sample(filters=128)
        self.down3 = Down_Sample(filters=256)
        self.down4 = Down_Sample(filters=512)

        self.con = Con_Bn_Act(filters=1024)

        self.out = Conv2D(filters=1,
                          kernel_size=3,
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
        self.down = Con_Bn_Act(filters=filters, strides=2)

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
                                             use_bias=False)
        self.bn = BatchNormalization()
        self.act = ReLU()

    def call(self, inputs, training=None, mask=None):
        con_transpose = self.con_transpose(inputs)
        bn = self.bn(con_transpose)
        out = self.act(bn)

        return out


class Vgg19_Discriminator(tf.keras.Model):
    def __init__(self, input_shape):
        super(Vgg19_Discriminator, self).__init__()
        m_tf = tf.keras.applications.vgg19.VGG19(include_top=False, input_shape=input_shape)
        layer_ids = [2, 5, 8, 13, 18]

        base_model_outputs = [m_tf.layers[layer_id].output for layer_id in layer_ids]
        self.model = tf.keras.Model(inputs=m_tf.input, outputs=base_model_outputs)
        self.model.trainable = False

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)


class Multi_Discriminator(Model):
    def __init__(self):
        super(Multi_Discriminator, self).__init__()

        self.down1 = Down_Sample(filters=64)
        self.down2 = Down_Sample(filters=128)
        self.down2_con = CBR_Block(filters=128, num_cbr=2)
        self.down2_out = Conv2D(filters=1, kernel_size=3, padding='same')

        self.down3 = Down_Sample(filters=256)
        self.down3_con = CBR_Block(filters=256, num_cbr=2)
        self.down3_out = Conv2D(filters=1, kernel_size=3, padding='same')

        self.down4 = Down_Sample(filters=512)
        self.down4_con = CBR_Block(filters=512, num_cbr=2)
        self.down4_out = Conv2D(filters=1, kernel_size=3, padding='same')

        self.down5 = Down_Sample(filters=1024)
        self.con = Con_Bn_Act(filters=1024)

        self.out = Conv2D(filters=1, kernel_size=3, padding='same')

    def call(self, inputs, training=None, mask=None):
        [inp, tar] = inputs
        concat = concatenate([inp, tar], axis=3)
        down1 = self.down1(concat)
        down2 = self.down2(down1)
        down2_con = self.down2_con(down2)
        down2_out = self.down2_out(down2_con)

        down3 = self.down3(down2)
        down3_con = self.down3_con(down3)
        down3_out = self.down3_out(down3_con)

        down4 = self.down4(down3)
        down4_con = self.down4_con(down4)
        down4_out = self.down4_out(down4_con)

        down5 = self.down5(down4)
        con = self.con(down5)
        out = self.out(con)

        return [down2_out, down3_out, down4_out, out]


class MultiscaleDiscriminator(Model):
    def __init__(self, filters=64):
        super(MultiscaleDiscriminator, self).__init__()
        self.filters = filters

        self.pooling = MaxPooling2D(padding='same')
        self.discriminator1 = NLayerDiscriminator(filters=self.filters)
        self.discriminator2 = NLayerDiscriminator(filters=self.filters)
        self.discriminator3 = NLayerDiscriminator(filters=self.filters)
        self.discriminator4 = NLayerDiscriminator(filters=self.filters)

    def call(self, inputs, training=None, mask=None):
        [inp, tar] = inputs
        inputs = concatenate([inp, tar], axis=3)

        discriminator1 = self.discriminator1(inputs)

        inputs = self.pooling(inputs)
        discriminator2 = self.discriminator2(inputs)

        inputs = self.pooling(inputs)
        discriminator3 = self.discriminator3(inputs)

        inputs = self.pooling(inputs)
        discriminator4 = self.discriminator4(inputs)

        return [discriminator1, discriminator2, discriminator3, discriminator4]


class NLayerDiscriminator(Model):
    def __init__(self, layers_num=3, filters=64):
        super(NLayerDiscriminator, self).__init__()
        self.layers_num = layers_num
        self.filters = filters

        self.cbr1 = Con_Bn_Act(filters=self.filters, strides=2)
        self.cbr2 = Con_Bn_Act(filters=self.filters * 2, strides=2)
        self.cbr3 = Con_Bn_Act(filters=self.filters * 4, strides=2)
        self.cbr4 = Con_Bn_Act(filters=self.filters * 8, strides=2, activation='not')

    def call(self, inputs, training=None, mask=None):
        cbr1 = self.cbr1(inputs)
        cbr2 = self.cbr2(cbr1)
        cbr3 = self.cbr3(cbr2)
        out = self.cbr4(cbr3)

        return out
