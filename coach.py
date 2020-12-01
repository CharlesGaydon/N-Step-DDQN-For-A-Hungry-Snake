# Imports
from tqdm import tqdm

from TwoPlayerSnake.TwoPlayerSnakeArena import TwoPlayerSnakeArena

# TODO : remove this
from utils import dotdict

args = dotdict(
    {
        "numIters": 2,
        "numEps": 3,  # Number of complete self-play games to simulate during a new iteration.
        "tempThreshold": 5,  # Threshold for making greedy actions in the self-play games
        "updateThreshold": 0.55,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        "maxlenOfQueue": 200,  # Number of game examples to train the neural networks.
        "numMCTSSims": 3,  # Number of games moves for MCTS to simulate.
        "arenaCompare": 7,  # Number of games to play during arena play to determine if new net will be accepted.
        "cpuct": 1,
        "checkpoint": "./temp/",
        "load_model": False,
        "load_folder_file": ("./tictactoe/trained/", "my_checkpoint"),
        "numItersForTrainExamplesHistory": 20,
    }
)


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

        for iter_ in tqdm(self.args.numIters, desc="Iterations"):
            for self_play_ in tqdm(self.args.numIters, desc="Self play"):
                arena.play_game()
                self.train_examples_history += arena.train_examples
                # TODO: test for size of history and update in consequence
                # TODO: change this selection of examples
                self.train_examples_history = self.train_examples_history[-1000:]

            # train the new model
            self.nnet.train(self.train_examples_history)

            # compare to the previous one
            wins, draws, loss = arena.play_games(self.args.arenaCompare)
            if (
                wins + loss == 0
                or float(wins) / (wins + loss) < self.args.updateThreshold
            ):
                # reject the model
                self.nnet.load_checkpoint(
                    folder=self.args.checkpoint, filename="temp.pth.tar"
                )
            else:
                # accept the model
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename=f"checkpoint_{iter_}.pth.tar"
                )
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename="best.pth.tar"
                )
