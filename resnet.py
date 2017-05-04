"""
Implementation of Deep Residual Network.

References:
  [1] "Deep Residual Learning for Image Recognition" https://arxiv.org/pdf/1512.03385.pdf
"""
from keras import backend as K
from keras.layers import Activation, Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add
from keras.models import Model
from keras.utils import plot_model


class ResidualUnit:
    """
    Residual unit as described in [1].
    """
    def __init__(self, filters, first_conv_strides):
        self.filters = filters
        self.first_conv_strides = first_conv_strides

    def __call__(self, x):
        conv1 = Conv2D(filters=self.filters, kernel_size=(3, 3),
                       strides=self.first_conv_strides, padding='same',
                       kernel_initializer='glorot_normal')(x)
        norm1 = BatchNormalization(axis=3)(conv1)
        relu1 = Activation('relu')(norm1)
        conv2 = Conv2D(filters=self.filters, kernel_size=(3, 3),
                       strides=(1, 1), padding='same',
                       kernel_initializer='glorot_normal')(relu1)
        norm2 = BatchNormalization(axis=3)(conv2)
        return Activation('relu')(self.shortcut_and_add(x, norm2))

    def shortcut_and_add(self, x, residual):
        x_shape = K.int_shape(x)
        residual_shape = K.int_shape(residual)
        shortcut = x
        if x_shape != residual_shape:
            conv1 = Conv2D(filters=residual_shape[3], kernel_size=(1, 1),
                           strides=self.first_conv_strides, padding='same',
                           kernel_initializer='glorot_normal')(x)
            shortcut = BatchNormalization(axis=3)(conv1)
        return add([shortcut, residual])


class BottleneckResidualUnit(ResidualUnit):
    """
    Bottleneck residual unit as described in [1] for ResNet-50/101/152.
    """

    def __call__(self, x):
        conv1 = Conv2D(filters=self.filters, kernel_size=(1, 1),
                       strides=self.first_conv_strides, padding='same',
                       kernel_initializer='glorot_normal')(x)
        norm1 = BatchNormalization(axis=3)(conv1)
        relu1 = Activation('relu')(norm1)
        conv2 = Conv2D(filters=self.filters, kernel_size=(3, 3),
                       strides=(1, 1), padding='same',
                       kernel_initializer='glorot_normal')(relu1)
        norm2 = BatchNormalization(axis=3)(conv2)
        relu2 = Activation('relu')(norm2)
        conv3 = Conv2D(filters=self.filters * 4, kernel_size=(3, 3),
                       strides=(1, 1), padding='same',
                       kernel_initializer='glorot_normal')(relu2)
        norm3 = BatchNormalization(axis=3)(conv3)
        return Activation('relu')(self.shortcut_and_add(x, norm3))


class ResidualBlock:
    def __init__(self, units, filters, residual_unit_cls, is_first_block=False):
        self.filters = filters
        self.units = units
        self.is_first_block = is_first_block
        self.residual_unit_cls = residual_unit_cls

    def __call__(self, x):
        current = x
        for i in range(self.units):
            strides = (1, 1)
            if not self.is_first_block and i == 0:
                strides = (2, 2)
            current = self.residual_unit_cls(
                filters=self.filters, first_conv_strides=strides)(current)
        return current


class IdentityResidualUnit:
    # TODO
    pass


class WideResidualUnit:
    # TODO
    pass


class ResnetFactory:
    @staticmethod
    def get_original(input_shape, num_classes, residual_unit_cls, units_per_block):
        """As described in [1]"""
        x = Input(shape=input_shape)
        conv1 = Conv2D(filters=64, kernel_size=(7, 7),
                       strides=(2, 2), padding='same',
                       kernel_initializer='glorot_normal')(x)
        norm1 = BatchNormalization(axis=3)(conv1)
        relu1 = Activation('relu')(norm1)
        current = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(relu1)
        filters = 64
        for i, units in enumerate(units_per_block):
            current = ResidualBlock(units, filters, residual_unit_cls, is_first_block=(i == 0))(current)
            filters *= 2
        relu1 = Activation('relu')(current)
        avg_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(relu1)
        flatten1 = Flatten()(avg_pool)
        dense = Dense(units=num_classes, activation='softmax')(flatten1)
        return Model(inputs=x, outputs=dense)

    @staticmethod
    def get_original_18(input_shape, num_classes):
        """As described in [1]"""
        return ResnetFactory.get_original(input_shape, num_classes, ResidualUnit, [2, 2, 2, 2])

    @staticmethod
    def get_original_34(input_shape, num_classes):
        """As described in [1]"""
        return ResnetFactory.get_original(input_shape, num_classes, ResidualUnit, [3, 4, 6, 3])

    @staticmethod
    def get_original_50(input_shape, num_classes):
        """As described in [1]"""
        return ResnetFactory.get_original(input_shape, num_classes, BottleneckResidualUnit, [3, 4, 6, 3])

    @staticmethod
    def get_original_101(input_shape, num_classes):
        """As described in [1]"""
        return ResnetFactory.get_original(input_shape, num_classes, BottleneckResidualUnit, [3, 4, 23, 3])

    @staticmethod
    def get_original_152(input_shape, num_classes):
        """As described in [1]"""
        return ResnetFactory.get_original(input_shape, num_classes, BottleneckResidualUnit, [3, 8, 36, 3])


if __name__ == '__main__':
    resnet34 = ResnetFactory.get_original_34((224, 224, 3), 1000)
    plot_model(resnet34, to_file='resnet_34png')
    resnet101 = ResnetFactory.get_original_101((224, 224, 3), 1000)
    plot_model(resnet101, to_file='resnet_101.png')
