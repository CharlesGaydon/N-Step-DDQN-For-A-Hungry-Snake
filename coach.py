# Imports
from tqdm import tqdm
import numpy as np
from TwoPlayerSnake.TwoPlayerSnakeArena import TwoPlayerSnakeArena


class Coach:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.train_examples_history = []

    def learn(self):

        arena = TwoPlayerSnakeArena(self.nnet, self.pnet, self.game, self.args)

        for iter_ in tqdm(range(self.args.numIters), desc="Iterations"):
            arena.deep_q_learning()

            # compare to the previous one
            # TODO: add logs of proportion of results
            wins, draws, loss, stats = arena.compare_two_models_n_times(
                self.args.arenaCompare, verbose=True
            )
            if (wins + loss == 0 and self.args.updateThreshold > 0) or float(wins) / (
                wins + loss
            ) < self.args.updateThreshold:
                print("Reject the model.")
                # reject the NNets
                self.nnet.set_weights(self.pnet)
            else:
                # accept the NNets
                print("Accept the model.")
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename=f"checkpoint_{iter_}.hdf5"
                )
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename="best.hdf5"
                )
                self.pnet.set_weights(self.nnet)
