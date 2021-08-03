import datetime
import tensorflow as tf
from loss.loss import discriminator_loss, generator_loss
from model.discriminator import Discriminator
from model.generator import Generator
from data_utils.data_loader import Data_Loader
# from tensorflow.keras import mixed_precision


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
#
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)


class Pix2pix_Trainer:
    def __init__(self, ex_name, epochs, batch_size, checkpoint_dir, data_size, load_weights):
        self.ex_name = ex_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.data_size = data_size
        self.load_weights = load_weights

        self.data_loader = Data_Loader(batch_size=self.batch_size, size=self.data_size)
        self.train_datasets = self.data_loader.get_train_datasets()
        self.val_datasets = self.data_loader.get_val_datasets()
        self.generator = Generator()
        self.discriminator = Discriminator()

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
        self.log_dir = './log/'
        self.summary_writer = tf.summary.create_file_writer(
            self.log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    def train(self):
        if self.load_weights and self.ck_manager.latest_checkpoint:
            self.checkpoint.restore(self.ck_manager.latest_checkpoint)
            print("[info]Restored from {}".format(self.ck_manager.latest_checkpoint))
        else:
            print("[info]Initializing from scratch.")

        for step, (input_image, target) in self.train_datasets.repeat().take(self.epochs).enumerate():
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = self.train_step(input_image, target, step)
            if (step + 1) % 100 == 0:
                print(f"Step: {step // 100}h"
                      + '\tgen_total_loss:' + str(gen_total_loss)
                      + '\tgen_gan_loss:' + str(gen_gan_loss)
                      + '\tgen_l1_loss:' + str(gen_l1_loss)
                      + '\tdisc_loss:' + str(disc_loss))
            if (step + 1) % 1000 == 0:
                self.ck_manager.save()

    @tf.function
    def train_step(self, input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 1000)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step // 1000)

        return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss


def main():
    ex_name = 'pix2pix_256'
    checkpoint_dir = './checkpoints/pix2pix_checkpoints/'

    start_time = datetime.datetime.now()
    trainer = Pix2pix_Trainer(ex_name=ex_name,
                              epochs=200*1000,
                              batch_size=32,
                              checkpoint_dir=checkpoint_dir,
                              data_size=256,
                              load_weights=True)
    trainer.train()

    end_time = datetime.datetime.now()
    print('time:\t' + str(end_time - start_time).split('.')[0])


if __name__ == '__main__':
    main()
