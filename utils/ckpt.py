import tensorflow as tf


class Checkpoint_Utils:
    def __init__(self,
                 checkpoint_directory,
                 checkpoint_name,
                 max_to_keep,
                 generator_optimizer,
                 discriminator_optimizer,
                 generator,
                 discriminator):
        self.max_to_keep = max_to_keep
        self.checkpoint_name = checkpoint_name
        self.checkpoint_directory = checkpoint_directory
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator = generator
        self.discriminator = discriminator

        self.ck_manager = tf.train.CheckpointManager(self.checkpoint,
                                                     directory=self.checkpoint_directory,
                                                     max_to_keep=self.max_to_keep,
                                                     checkpoint_name=self.checkpoint_name + '_ck')
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    def get_ckpt_manager(self):
        return self.checkpoint, self.ck_manager
