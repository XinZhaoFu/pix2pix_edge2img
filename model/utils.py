from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras import Model, regularizers


class Con_Bn_Act(Model):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 padding='same',
                 strides=1,
                 activation=None,
                 dilation_rate=1,
                 name=None,
                 kernel_regularizer=False,
                 kernel_initializer=None,
                 train_able=True):
        super(Con_Bn_Act, self).__init__()
        self.kernel_regularizer = kernel_regularizer
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.dilation_rate = dilation_rate
        self.block_name = name
        self.kernel_initializer = kernel_initializer
        self.train_able = train_able

        if self.activation is None:
            self.activation = 'relu'

        if self.kernel_initializer is None:
            self.kernel_initializer = 'glorot_uniform'

        if self.kernel_regularizer:
            self.kernel_regularizer = regularizers.l2()
        else:
            self.kernel_regularizer = None

        # kernel_initializer_special_cases = ['glorot_uniform',
        # 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']
        self.con = Conv2D(filters=self.filters,
                          kernel_size=self.kernel_size,
                          padding=self.padding,
                          strides=self.strides,
                          use_bias=False,
                          dilation_rate=(self.dilation_rate, self.dilation_rate),
                          name=self.block_name,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_initializer=self.kernel_initializer)

        if self.train_able is False:
            self.con.trainable = False

        self.bn = BatchNormalization()

        if self.activation != 'not':
            self.act = Activation(self.activation)

    def call(self, inputs, training=None, mask=None):
        con = self.con(inputs)
        bn = self.bn(con)
        if self.kernel_size == (1, 1) or self.activation == 'not':
            out = bn
        else:
            out = self.act(bn)
        return out
