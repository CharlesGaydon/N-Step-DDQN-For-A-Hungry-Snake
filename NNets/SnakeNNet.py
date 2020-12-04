# from https://github.com/suragnair/alpha-zero-general/

# import sys
# sys.path.append('..')
import argparse
from keras.models import *
from keras.layers import *
from keras.optimizers import *


class SnakeNNet:
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.get_board_dimensions()
        self.action_size = game.action_size
        self.args = args

        pool_x = self.board_x // 5 + 1
        pool_y = self.board_y // 5 + 1

        # Neural Net
        self.input_boards = Input(
            shape=(self.board_x, self.board_y)
        )  # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x, self.board_y, 1))(
            self.input_boards
        )  # batch_size  x board_x x board_y x 1
        mpool_1 = MaxPooling2D(
            pool_size=(pool_x + 1, pool_y + 1), strides=(1, 1), padding="valid"
        )
        h_conv1 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(args.num_channels, 3, padding="same")(x_image)
            )
        )  # batch_size  x board_x x board_y x num_channels
        x_2 = mpool_1(h_conv1)

        mpool_2 = MaxPooling2D(
            pool_size=(pool_x, pool_y), strides=(1, 1), padding="valid"
        )
        h_conv2 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(args.num_channels, 3, padding="same")(x_2)
            )
        )  # batch_size  x board_x x board_y x num_channels
        x_2 = mpool_2(h_conv2)

        mpool_3 = MaxPooling2D(
            pool_size=(pool_x, pool_y), strides=(1, 1), padding="valid"
        )
        h_conv3 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(args.num_channels, 3, padding="same")(x_2)
            )
        )  # batch_size  x board_x x board_y x num_channels
        x_3 = mpool_3(h_conv3)

        mpool_4 = MaxPooling2D(
            pool_size=(pool_x - 1, pool_y - 1), strides=(1, 1), padding="valid"
        )
        h_conv4 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(args.num_channels * 2, 3, padding="same")(x_3)
            )
        )  # batch_size  x board_x x board_y x num_channels
        x_4 = mpool_4(h_conv4)

        h_conv_flat = Flatten()(x_4)

        s_fc1 = Dropout(args.dropout)(
            Activation("relu")(
                BatchNormalization(axis=1)(Dense(args.num_channels * 2)(h_conv_flat))
            )
        )  # batch_size x xx
        s_fc2 = Dropout(args.dropout)(
            Activation("relu")(
                BatchNormalization(axis=1)(Dense(args.num_channels)(s_fc1))
            )
        )  # batch_size x xx
        self.q_a = Dense(self.action_size, activation="tanh", name="q_a")(
            s_fc2
        )  # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=self.q_a)
        self.model.compile(loss=["mean_squared_error"], optimizer=Adam(args.lr))
