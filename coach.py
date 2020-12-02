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
        self.train_examples_history = (
            []
        )  # history of examples from args.numItersForTrainExamplesHistory latest iterations

    def learn(self):

        arena = TwoPlayerSnakeArena(self.nnet, self.pnet, self.game)

        for iter_ in tqdm(range(self.args.numIters), desc="Iterations"):
            for self_play_ in tqdm(range(self.args.numIters), desc="Self play"):
                arena.play_game(keep_track_of_historic=True, display=False)
                self.train_examples_history += arena.train_examples
                # TODO: test for size of history and update in consequence
                # TODO: change this selection of examples
                self.train_examples_history = self.train_examples_history[-3000:]

            # train the new NNets
            print(f"Training on {len(self.train_examples_history)} examples.")
            self.nnet.train(self.train_examples_history)

            # compare to the previous one
            # TODO: add logs of proportion of results
            wins, draws, loss, stats = arena.compare_two_models(
                self.args.arenaCompare, verbose=True
            )
            if (
                wins + loss == 0
                or float(wins) / (wins + loss) < self.args.updateThreshold
            ):
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
