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
        self.board_x, self.board_y = game.get_board_dimensions()
        # models
        self.nnet = snet(game, args)
        self.target_nnet = snet(game, args)
        self.update_target_nnet()
        self.optimization_step_taken = 0

    def optimize_network(self, experiences, args):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        sample_idx = np.random.choice(len(experiences), args.batch_size)
        sample = []
        for idx in sample_idx:
            sample.append(experiences[idx])

        X_batch = []
        Y_batch = []

        for last_s, last_a, r, terminal, s in sample:
            q_values_target = self.predict_action_values_from_state(
                last_s, use_target_nnet=True
            )
            if terminal:
                target_q_s_a = r
            else:
                max_q_value = np.max(
                    self.predict_action_values_from_state(s, use_target_nnet=False)
                )
                target_q_s_a = r + args.discount_factor * max_q_value
            q_values_target[last_a] = target_q_s_a
            # fill batch
            X_batch.append(last_s)
            Y_batch.append(q_values_target)

        X_batch = np.array(X_batch, dtype=np.float64)
        Y_batch = np.array(Y_batch, dtype=np.float64)
        self.nnet.model.fit(
            X_batch, Y_batch, batch_size=args.batch_size, epochs=1, verbose=0
        )

    def predict_action_values_from_game(self, game, perspective=None):
        """
        game: snake game
        perspective : 1 or 2 depending on the considered player
        """

        # preparing input
        board = game.get_board(perspective)
        board = board[np.newaxis, :, :]

        # run
        v_pred = self.nnet.model.predict(board)

        return v_pred[0]

    def predict_action_values_from_state(self, board, use_target_nnet):
        if use_target_nnet:
            current_model = self.target_nnet.model
        else:
            current_model = self.nnet.model
        board = board[np.newaxis, :, :]  # useful ?
        v_pred = current_model.predict(board)

        return v_pred[0]

    def epsilon_greedy_policy(self, game, args, perspective=None):
        """
        game: snake game
        perspective : 1 or 2 depending on the considered player
        """
        # todo: add decaying epsilon
        if np.random.random() < args.epsilon:
            return np.random.choice(4)
        else:
            return self.greedy_policy(game, args, perspective=perspective)

    def greedy_policy(self, game, args, perspective=None):
        """
        game: snake game
        perspective : 1 or 2 depending on the considered player
        """

        q = self.predict_action_values_from_game(game, perspective=perspective)
        # todo: replace by a nice regularized softmax here, with tau parameter for temperature
        q = np.exp(q / args.temperature)
        q = q / q.sum()
        # select optimal action
        a = np.random.choice(range(4), p=q)

        return a

    def set_weights(self, other_nnet_wrapper):
        self.nnet.model.set_weights(other_nnet_wrapper.nnet.model.get_weights())

    def update_target_nnet(self):
        self.target_nnet.model.set_weights(self.nnet.model.get_weights())

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
