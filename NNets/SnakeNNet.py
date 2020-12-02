# from https://github.com/suragnair/alpha-zero-general/

# import sys
# sys.path.append('..')
import argparse
from keras.models import *
from keras.layers import *
from keras.optimizers import *

"""

"""


class SnakeNNet:
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.get_board_dimensions()
        self.action_size = 4
        self.args = args

        # Neural Net
        self.input_boards = Input(
            shape=(self.board_x, self.board_y)
        )  # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x, self.board_y, 1))(
            self.input_boards
        )  # batch_size  x board_x x board_y x 1
        h_conv1 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(args.num_channels, 3, padding="same")(x_image)
            )
        )  # batch_size  x board_x x board_y x num_channels
        h_conv1_flat = Flatten()(h_conv1)
        s_fc1 = Dropout(args.dropout)(
            Activation("relu")(
                BatchNormalization(axis=1)(
                    Dense((self.board_x * self.board_y) // 2)(h_conv1_flat)
                )
            )
        )  # batch_size x xx
        s_fc2 = Dropout(args.dropout)(
            Activation("relu")(
                BatchNormalization(axis=1)(
                    Dense((self.board_x * self.board_y) // 4)(s_fc1)
                )
            )
        )  # batch_size x xx
        self.pi = Dense(self.action_size, activation="softmax", name="pi")(
            s_fc2
        )  # batch_size x self.action_size
        self.v = Dense(1, activation="tanh", name="v")(s_fc2)  # batch_size x 1
        # TODO: understand why we train on predicting pi!
        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        # TODO: add an early stopping here to avoid overfitting
        self.model.compile(
            loss=["categorical_crossentropy", "mean_squared_error"],
            optimizer=Adam(args.lr),
        )
