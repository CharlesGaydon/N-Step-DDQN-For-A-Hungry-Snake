# Imports
from tqdm import tqdm
from .OnePlayerSnakeArena import OnePlayerSnakeArena


class Coach:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

    def learn(self):

        arena = OnePlayerSnakeArena(self.nnet, self.game, self.args)

        for iter_ in tqdm(range(self.args.numIters), desc="Iterations"):

            arena.deep_q_learning()
            self.nnet.save_checkpoint(
                folder=self.args.load_folder_file[0],
                filename=f"checkpoint_{iter_}.hdf5",
            )
            self.nnet.save_checkpoint(
                folder=self.args.load_folder_file[0],
                filename=self.args.load_folder_file[1],
            )
            # compare to the previous one
            arena.play_n_games(self.args.arenaCompare, verbose=True)
