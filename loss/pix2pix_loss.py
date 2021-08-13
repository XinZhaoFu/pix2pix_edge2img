import tensorflow as tf
from model.pix2pix import Vgg19_Discriminator


def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target, loss_lambda=100):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    down2_out, down3_out, down4_out, out = disc_generated_output

    gan_loss = loss_object(tf.ones_like(out), out)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (loss_lambda * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


class VGG19_Discriminator_Loss(tf.keras.losses.Loss):
    def __init__(self, input_shape=None):
        super(VGG19_Discriminator_Loss, self).__init__()
        if input_shape is None:
            input_shape = [512, 512, 3]
        self.vgg = Vgg19_Discriminator(input_shape=input_shape)
        self.criterion = tf.keras.losses.MeanAbsoluteError()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def call(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(tf.stop_gradient(y_vgg[i]), x_vgg[i])
        return loss


def dice_bce_loss():
    bce = tf.keras.losses.binary_crossentropy
    smooth = 1.0

    def dice_bce_loss_fixed(y_true, y_pred):
        y_true_sum = tf.reduce_sum(y_true)
        y_pred_sum = tf.reduce_sum(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)

        dice_loss = 1 - (2.0 * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)
        bce_loss = bce(y_true, y_pred)
        loss = dice_loss * 0.5 + bce_loss * 0.5

        return loss

    return dice_bce_loss_fixed


def multi_discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_down2_out, real_down3_out, real_down4_out, real_out = disc_real_output
    generated_down2_out, generated_down3_out, generated_down4_out, generated_out = disc_generated_output

    real_down2_out_loss = loss_object(tf.ones_like(real_down2_out), real_down2_out)
    real_down3_out_loss = loss_object(tf.ones_like(real_down3_out), real_down3_out)
    real_down4_out_loss = loss_object(tf.ones_like(real_down4_out), real_down4_out)
    real_out_loss = loss_object(tf.ones_like(real_out), real_out)

    real_loss = real_down2_out_loss * 1. / 8 + real_down3_out_loss * 1. / 4 + \
                real_down4_out_loss * 1. / 2 + real_out_loss * 1.

    generated_down2_out_loss = loss_object(tf.zeros_like(generated_down2_out), generated_down2_out)
    generated_down3_out_loss = loss_object(tf.zeros_like(generated_down3_out), generated_down3_out)
    generated_down4_out_loss = loss_object(tf.zeros_like(generated_down4_out), generated_down4_out)
    generated_out_loss = loss_object(tf.zeros_like(generated_out), generated_out)

    generated_loss = generated_down2_out_loss * 1. / 8 + generated_down3_out_loss * 1. / 4 + \
                     generated_down4_out_loss * 1. / 2 + generated_out_loss * 1.
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss
