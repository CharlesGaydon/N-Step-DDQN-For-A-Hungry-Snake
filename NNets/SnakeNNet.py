# from https://github.com/suragnair/alpha-zero-general/

# import sys
# sys.path.append('..')
import argparse
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras.initializers import VarianceScaling


class SnakeNNet:
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.get_board_dimensions()
        self.action_size = game.action_size
        self.args = args

        # Neural Net
        initializer = VarianceScaling(
            scale=1.0, mode="fan_in", distribution="normal", seed=None
        )
        model = Sequential()
        model.add(
            Reshape(
                (self.board_x, self.board_y, 1),
                input_shape=(self.board_x, self.board_y),
            )
        )  # batch_size  x board_x x board_y x 1)

        model.add(
            Convolution2D(
                8,
                (3, 3),
                strides=(1, 1),
                padding="same",
                kernel_initializer=initializer,
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(
            Convolution2D(
                16,
                (3, 3),
                strides=(2, 2),
                padding="same",
                kernel_initializer=initializer,
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(
            Convolution2D(
                32,
                (3, 3),
                strides=(2, 2),
                padding="same",
                kernel_initializer=initializer,
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))

        model.add(Flatten())
        model.add(Dense(16, kernel_initializer=initializer))
        model.add(Dense(self.action_size, kernel_initializer=initializer))
        adam = Adam(lr=args.learning_rate, clipnorm=1.0)
        model.compile(loss=[self.masked_loss_function], optimizer=adam)

        self.model = model
        print(self.model.summary())
        # exit()

        # FOR BIGGER
        # model.add(Convolution2D(16, (4, 4), strides=(2, 2), padding="same", kernel_initializer=initializer))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=-1))
        # model.add(Convolution2D(16, (4, 4), strides=(2, 2), padding="same", kernel_initializer=initializer))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=-1))
        # model.add(Convolution2D(16, (3, 3), strides=(2, 2), padding="same", kernel_initializer=initializer))
        # model.add(Activation("relu"))

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)

    def masked_loss_function(self, y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, self.args.mask_value), K.floatx())
        return self.mean_squared_error(y_true * mask, y_pred * mask)
