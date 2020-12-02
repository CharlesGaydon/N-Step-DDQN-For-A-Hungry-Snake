from TwoPlayerSnake.TwoPlayerSnakeGame import TwoPlayerSnakeGame
from coach import Coach
from NNets.NNet import NNetWrapper

from utils import dotdict


def main():
    g = TwoPlayerSnakeGame(board_x=10, board_y=10)
    nnet = NNetWrapper(g)
    args = dotdict(
        {
            "numIters": 2,
            "numEps": 3,  # Number of complete self-play games to simulate during a new iteration.
            "tempThreshold": 5,  # Threshold for making greedy actions in the self-play games
            "updateThreshold": 0.55,
            # During arena playoff, new neural net will be accepted if threshold or more of games are won.
            "maxlenOfQueue": 200,  # Number of game examples to train the neural networks.
            "numMCTSSims": 3,  # Number of games moves for MCTS to simulate.
            "arenaCompare": 7,  # Number of games to play during arena play to determine if new net will be accepted.
            "cpuct": 1,
            "checkpoint": "./NNEts/trained/",
            "load_model": False,
            "load_folder_file": ("./NNets/trained/", "best_checkpoint"),
            "numItersForTrainExamplesHistory": 20,
        }
    )
    c = Coach(g, nnet, args)
    c.learn()

    return 0


if __name__ == "__main__":
    main()
