# from https://github.com/suragnair/alpha-zero-general/

import time
import numpy as np
import sys

sys.path.append("..")
from utils import dotdict
from .SnakeNNet import SnakeNNet as snet

"""
NeuralNet wrapper class for the TicTacToeNNet.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on (copy-pasted from) the NNet by SourKream and Surag Nair.
"""

args = dotdict(
    {
        "lr": 0.001,
        "dropout": 0.3,
        "epochs": 2,
        "batch_size": 64,
        "cuda": False,
        "num_channels": 15,
    }
)


class NNetWrapper:
    def __init__(self, game):
        self.nnet = snet(game, args)
        self.board_x, self.board_y = game.get_board_dimensions()
        self.action_size = 4

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        # Todo: here select only a subset of examples for efficiency ! Recall
        # Here only the last 1000 examples are used !!!

        input_boards, target_pis, target_vs = list(zip(*examples[-1000:]))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(
            x=input_boards,
            y=[target_pis, target_vs],
            batch_size=args.batch_size,
            epochs=args.epochs,
        )

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(
        self, folder="./keras/checkpoint", filename="checkpoint.pth.tar"
    ):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(
                    folder
                )
            )
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(
        self, folder="./keras/checkpoint", filename="checkpoint.pth.tar"
    ):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        print(filepath)
        if not os.path.exists(filepath):
            raise ValueError("No model in path '{}'".format(filepath))
        self.nnet.model.load_weights(filepath)
