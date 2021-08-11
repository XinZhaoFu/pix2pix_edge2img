# coding=utf-8
import os
import datetime
import tensorflow as tf
from model.unet import UNet
from tensorflow.keras import mixed_precision
from data_utils.img2img_data_loader import Data_Loader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print('[INFO] 计算类型: %s' % policy.compute_dtype)
print('[INFO] 变量类型: %s' % policy.variable_dtype)


class train:
    def __init__(self):
        self.batch_size = 4
        self.data_size = 512
        self.load_weights = True
        self.checkpoint_save_path = './checkpoints/unet512_64/unet512.ckpt'
        self.checkpoint_input_path = self.checkpoint_save_path
        self.epochs = 1000

        self.data_loader = Data_Loader(batch_size=self.batch_size, size=self.data_size)
        self.train_datasets = self.data_loader.get_train_datasets()
        self.val_datasets = self.data_loader.get_val_datasets()
        print(self.train_datasets)

    def model_train(self):
        """
        可多卡训练

        :return:
        """

        model = UNet(semantic_filters=64,
                     detail_filters=64,
                     output_channels=3,
                     semantic_num_cbr=1,
                     detail_num_cbr=4,
                     end_activation='sigmoid')

        model.compile(
            optimizer='Adam',
            loss='mse',
            metrics=['accuracy']
        )

        if os.path.exists(self.checkpoint_input_path + '.index') and self.load_weights:
            print("[INFO] -------------------------------------------------")
            print("[INFO] -----------------loading weights-----------------")
            print("[INFO] -------------------------------------------------")
            model.load_weights(self.checkpoint_input_path)

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_save_path,
            monitor='val_loss',
            save_weights_only=True,
            save_best_only=True,
            mode='auto',
            save_freq='epoch')

        history = model.fit(
            self.train_datasets,
            epochs=self.epochs,
            verbose=1,
            validation_data=self.val_datasets,
            validation_freq=1,
            callbacks=[checkpoint_callback]
        )

        if self.epochs == 1:
            # 一般都是训练前专门看一下信息 所以正常训练时就不显示了 主要还是tmux不能上翻 有的时候会遮挡想看的信息
            model.summary()

        return history


def train_init():
    start_time = datetime.datetime.now()

    seg = train()
    seg.model_train()

    end_time = datetime.datetime.now()
    print('time:\t' + str(end_time - start_time).split('.')[0])


if __name__ == '__main__':
    train_init()
