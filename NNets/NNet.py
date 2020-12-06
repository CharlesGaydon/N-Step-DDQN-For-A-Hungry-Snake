# from https://github.com/suragnair/alpha-zero-general/

import numpy as np
import sys
import os

sys.path.append("..")
from utils import dotdict
from .SnakeNNet import SnakeNNet as snet

"""
NeuralNet wrapper class for the TicTacToeNNet.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on (copy-pasted from) the NNet by SourKream and Surag Nair.
"""

nnet_args = dotdict(
    {
        "learning_rate": 0.001,
        "dropout": 0.1,
        "cuda": False,
        "num_channels": 15,
        "mask_value": 424242,
    }
)


class NNetWrapper:
    def __init__(self, game, load_folder_file=False):
        self.board_x, self.board_y = game.get_board_dimensions()
        # models
        self.nnet = snet(game, nnet_args)
        self.target_nnet = snet(game, nnet_args)
        self.update_target_nnet()
        self.optimization_step_taken = 0
        self.loss_historic = []
        if load_folder_file:
            self.load_checkpoint(
                folder=load_folder_file[0], filename=load_folder_file[1]
            )

        self.fit_is_called_counter = 0

    def optimize_network(self, experiences, args):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        mask_value = self.nnet.args.mask_value
        sample_idx = np.random.choice(len(experiences), args.batch_size)
        sample = []
        for idx in sample_idx:
            sample.append(experiences[idx])

        X_batch = []
        Y_batch = []

        for last_s, last_a, expected_return, terminal, s in sample:
            q_values_target = np.array([mask_value, mask_value, mask_value, mask_value])
            if terminal:
                target_q_s_a = expected_return
            else:
                max_q_value = np.max(
                    self.predict_action_values_from_state(s, use_target_nnet=True)
                )
                target_q_s_a = (
                    expected_return
                    + (args.discount_factor ** args.n_step_learning) * max_q_value
                )
            q_values_target[last_a] = target_q_s_a

            X_batch.append(last_s)
            Y_batch.append(q_values_target)

        X_batch = np.array(X_batch, dtype=np.float64)
        Y_batch = np.array(Y_batch, dtype=np.float64)
        history = self.nnet.model.fit(
            X_batch, Y_batch, batch_size=args.batch_size, epochs=1, verbose=False
        )
        self.fit_is_called_counter += 1
        loss = history.history["loss"][0]
        self.loss_historic.append(loss)
        if len(self.loss_historic) > 20:
            self.loss_historic.pop(0)  # remove the first one

    def predict_action_values_from_game(self, game, perspective=None):
        """
        game: snake game
        perspective : 1 or 2 depending on the considered player
        """

        # preparing input
        board = game.get_board(perspective=perspective)
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

    def epsilon_greedy_policy(
        self, game, args, perspective=None, forbidden_direction=False
    ):
        """
        game: snake game
        perspective : 1 or 2 depending on the considered player
        """
        # todo: add decaying epsilon
        if np.random.random() < args.epsilon:
            while True:
                a = np.random.choice(4)
                if forbidden_direction is None:
                    return a
                else:
                    if a != forbidden_direction:
                        return a
        else:
            return self.greedy_policy(
                game, perspective=perspective, forbidden_direction=forbidden_direction
            )

    def greedy_policy(self, game, perspective=None, forbidden_direction=None):
        """
        game: snake game
        perspective : 1 or 2 depending on the considered player
        """

        q = self.predict_action_values_from_game(game, perspective=perspective)

        # never take the direction that is not possible
        if forbidden_direction is not None:
            q[forbidden_direction] = q.min()
        a = np.argmax(q)

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
            os.makedirs(folder)
        else:
            print(f"Saving to {folder}/{filename}")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder="./NNets/trained", filename="checkpoint.hdf5"):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        print(filepath)
        if not os.path.exists(filepath):
            raise ValueError("No NNets in path '{}'".format(filepath))
        self.nnet.model.load_weights(filepath)
