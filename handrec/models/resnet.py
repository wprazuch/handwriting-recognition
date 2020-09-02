from tensorflow.keras.layers import (BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D,
 ZeroPadding2D, Activation, Dense, Flatten, Input, add)

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chan_dim, red=False, reg=0.0001, bn_eps=2e-5, bn_mom=0.9):
        shortcut = data

        bn1 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

        bn2 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding='same', use_bias=False, kernel_regularizer=l2(reg))(act2)

        bn3 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)
        
        x = add([conv3, shortcut])

        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=0.0001, bn_eps=2e-5, bn_mom=0.9, dataset='cifar'):

        input_shape = (height, width, depth)
        chan_dim = -1

        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chan_dim = 1
        
        inputs = Input(shape=input_shape)
        x = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(inputs)
        x = Conv2D(filters[0], (3, 3), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)


        for i in range(0, len(stages)):
			# initialize the stride, then apply a residual module
			# used to reduce the spatial size of the input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride,
                chan_dim, red=True, bn_eps=bn_eps, bn_mom=bn_mom)

            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a ResNet module
                x = ResNet.residual_module(x, filters[i + 1],
                    (1, 1), chan_dim, bn_eps=bn_eps, bn_mom=bn_mom)

        # apply BN => ACT => POOL
        x = BatchNormalization(axis=chan_dim, epsilon=bn_eps,
            momentum=bn_mom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)

        # create the model
        model = Model(inputs, x, name="resnet")

        # return the constructed network architecture
        return model
