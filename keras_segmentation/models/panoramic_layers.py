from math import ceil

from keras.engine import Layer
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
import tensorflow as tf


class HorizontalRingPadding2D(Layer):

    def __init__(self, shape, kernel_size, strides, **kwargs):
        self.padding = self.compute_padding(shape, kernel_size, strides)
        super(HorizontalRingPadding2D, self).__init__(**kwargs)

    def build(self, input_shape):
        return super(HorizontalRingPadding2D, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        padded_shape = input_shape
        padded_pixels = self.padding

        width = padded_shape[2] + padded_pixels[0] + padded_pixels[1]
        height = padded_shape[1] + padded_pixels[2] + padded_pixels[3]

        padded_shape = padded_shape[0], height, width, padded_shape[3]

        return padded_shape

    def call(self, x, **kwargs):
        shape = x.get_shape().as_list()
        padded = tf.concat([x[:, 0:self.padding[0]], x, x[:, shape[1] - self.padding[1]:shape[1]]], axis=1)
        padded = tf.pad(padded, ((0, 0), (0, 0), (self.padding[2], self.padding[3]), (0, 0)), constant_values=0)

        return padded

    @staticmethod
    def compute_padding(shape, kernel_size, strides):
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)

        if type(strides) == int:
            strides = (strides, strides)

        out_height = ceil(float(shape[0]) / float(strides[0]))
        out_width = ceil(float(shape[1]) / float(strides[1]))

        pad_along_height = max((out_height - 1) * strides[0] + kernel_size[0] - shape[0], 0)
        pad_along_width = max((out_width - 1) * strides[1] + kernel_size[1] - shape[1], 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return pad_left, pad_right, pad_top, pad_bottom


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
        self.kernel_size = kernel_size
        self.strides = strides

        super(Conv2DPano, self).__init__()

    def build(self, input_shape):
        if self.padding == 'same':
            self.ring_padding = HorizontalRingPadding2D((input_shape[1], input_shape[2]), self.kernel_size, self.strides)

        return super(Conv2DPano, self).build(input_shape)

    def compute_output_shape(self, input_shape):

        if self.padding == 'same':
            padded_shape = self.ring_padding.compute_output_shape(input_shape)
        else:
            padded_shape = input_shape

        return self.conv.compute_output_shape(padded_shape)

    def call(self, x, **kwargs):
        padded = x

        if self.padding == 'same':
            padded = self.ring_padding(x)

        out = self.conv(padded, **kwargs)

        return out


class MaxPooling2DPano(Layer):

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', data_format=None, **kwargs):
        self.pool = MaxPooling2D(pool_size=pool_size, strides=strides, padding='valid',
                                 data_format=data_format, **kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

        super(MaxPooling2DPano, self).__init__()

    def build(self, input_shape):
        if self.padding == 'same':
            self.ring_padding = HorizontalRingPadding2D((input_shape[1], input_shape[2]), self.pool_size, self.strides)

        return super(MaxPooling2DPano, self).build(input_shape)

    def compute_output_shape(self, input_shape):

        if self.padding == 'same':
            padded_shape = self.ring_padding.compute_output_shape(input_shape)
        else:
            padded_shape = input_shape

        return self.pool.compute_output_shape(padded_shape)

    def call(self, x, **kwargs):
        padded = x

        if self.padding == 'same':
            padded = self.ring_padding(x)

        out = self.pool(padded, **kwargs)

        return out


class AveragePooling2DPano(Layer):

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', data_format=None, **kwargs):
        self.pool = AveragePooling2D(pool_size=pool_size, strides=strides, padding='valid',
                                 data_format=data_format, **kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

        super(AveragePooling2DPano, self).__init__()

    def build(self, input_shape):
        if self.padding == 'same':
            self.ring_padding = HorizontalRingPadding2D((input_shape[1], input_shape[2]), self.pool_size, self.strides)

        return super(AveragePooling2DPano, self).build(input_shape)

    def compute_output_shape(self, input_shape):

        if self.padding == 'same':
            padded_shape = self.ring_padding.compute_output_shape(input_shape)
        else:
            padded_shape = input_shape

        return self.pool.compute_output_shape(padded_shape)

    def call(self, x, **kwargs):
        padded = x

        if self.padding == 'same':
            padded = self.ring_padding(x)

        out = self.pool(padded, **kwargs)

        return out

