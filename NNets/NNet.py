# from https://github.com/suragnair/alpha-zero-general/

import time
import numpy as np
import sys
import os
from copy import deepcopy

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
        "epochs": 5,
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
        # TODO: add early stopping

        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(
            x=input_boards,
            y=[target_pis, target_vs],
            batch_size=args.batch_size,
            epochs=args.epochs,
        )

    def predict(self, game, perspective=None):
        """
        game: snake game
        perspective : 1 or 2 depending on the considered player
        """

        # preparing input
        board = game.get_board(perspective)
        board = board[np.newaxis, :, :]

        # run
        pi_pred, v_pred = self.nnet.model.predict(board)

        return pi_pred[0], v_pred[0]

    def make_policy_from_q_values(self, game, perspective=None):
        """
        game: snake game
        perspective : 1 or 2 depending on the considered player
        """
        # TODO long terme : replace by a kind of planning system ?
        q = np.zeros((4, 4))
        for a1 in range(4):
            for a2 in range(4):
                # générer le board résultant
                game_bis = deepcopy(game)
                game_bis.step(a1, a2)
                # évaluer ce board
                _, q[a1, a2] = self.predict(game_bis, perspective=perspective)
        q = q.mean(axis=1)
        q = q - q.min()
        q_mean = q.mean()
        q = q / q_mean
        return q, q_mean

    def set_weights(self, other_nnet_wrapper):
        self.nnet.model.set_weights(other_nnet_wrapper.nnet.model.get_weights())

    def save_checkpoint(self, folder="./NNets/checkpoint", filename="checkpoint.hdf5"):
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

    def load_checkpoint(self, folder="./NNets/trained", filename="checkpoint.hdf5"):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        print(filepath)
        if not os.path.exists(filepath):
            raise ValueError("No NNets in path '{}'".format(filepath))
        self.nnet.model.load_weights(filepath)
