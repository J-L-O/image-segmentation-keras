from keras.layers import Conv2D
from keras import backend as K


class Conv2DPano(Conv2D):

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1),
                 activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        super(Conv2DPano, self).__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                         data_format=data_format, dilation_rate=dilation_rate, activation=activation,
                                         use_bias=use_bias, kernel_initializer=kernel_initializer,
                                         bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                                         bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)

    def build(self, input_shape):
        return super(Conv2DPano, self).build(input_shape)

    def call(self, x):
        if self.padding == 'same':
            padded = K.spatial_2d_padding(x, padding=(1, 1), data_format=self.data_format)
        else:
            return super(Conv2DPano, self).call(x)
