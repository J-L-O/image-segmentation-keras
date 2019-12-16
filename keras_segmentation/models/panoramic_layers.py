from math import ceil

from keras.engine import Layer
from keras.layers import Conv2D
from keras import backend as K
import tensorflow as tf


class PanoPadding2D(Layer):

    def __init__(self, padding, **kwargs):
        self.padding = padding
        super(PanoPadding2D, self).__init__(**kwargs)

    def build(self, input_shape):
        return super(PanoPadding2D, self).build(input_shape)

    # def compute_output_shape(self, input_shape):
    #    return (input_shape[0])

    def call(self, x, **kwargs):
        shape = x.get_shape().as_list()
        padded = tf.concat([x[:, 0:self.padding[0]], x, x[:, shape[1] - self.padding[1]:shape[1]]], axis=1)
        padded = tf.pad(padded, ((0, 0), (0, 0), (self.padding[2], self.padding[3]), (0, 0)), constant_values=0)

        return padded


class Conv2DPano(Layer):

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1),
                 activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        self.conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='valid',
                           data_format=data_format, dilation_rate=dilation_rate, activation=activation,
                           use_bias=use_bias, kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)
        self.padding = padding
        self.pano_padding = PanoPadding2D(self._compute_padding((190, 385), kernel_size, strides))

        super(Conv2DPano, self).__init__()

    def build(self, input_shape):
        return super(Conv2DPano, self).build(input_shape)

    # def compute_output_shape(self, input_shape):
    #    return (input_shape[0])

    def _compute_padding(self, shape, kernel_size, strides):
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)

        if type(strides) == int:
            strides = (strides, strides)

        out_height = ceil(float(shape[0]) / float(strides[0]))
        out_width = ceil(float(shape[1]) / float(strides[1]))

        pad_along_height = max((out_height - 1) * strides[0] +
                               kernel_size[0] - shape[0], 0)
        pad_along_width = max((out_width - 1) * strides[1] +
                              kernel_size[1] - shape[1], 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return pad_left, pad_right, pad_top, pad_bottom

    def call(self, x, **kwargs):
        padded = x

        if self.padding == 'same':
            padded = self.pano_padding(x)

        return self.conv(padded, **kwargs)


class Conv2DPano2(Conv2D):

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1),
                 activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        super(Conv2DPano2, self).__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                         data_format=data_format, dilation_rate=dilation_rate, activation=activation,
                                         use_bias=use_bias, kernel_initializer=kernel_initializer,
                                         bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                                         bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)

    def build(self, input_shape):
        return super(Conv2DPano2, self).build(input_shape)

    def _compute_padding(self):
        return (0, 0, 0, 0)

    def call(self, x):
        if self.padding == 'same':
            shape = x.get_shape().as_list()
            padding = self._compute_padding()
            padded = tf.concat([x[:, 0:padding[0]], x, x[:, shape[1] - padding[1]:shape[1]]], axis=1)
            padded = tf.pad(padded, ((0, 0), (0, 0), (2, 2), (0, 0)), constant_values=0)

            self.padding = 'valid'

        return super(Conv2DPano2, self).call(x)
