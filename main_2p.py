from TwoPlayerSnake.TwoPlayerSnakeGame import TwoPlayerSnakeGame
from TwoPlayerSnake.TwoPlayerSnakeCoach import Coach
from NNets.NNet import NNetWrapper

from utils import dotdict

args = dotdict(
    {
        "numIters": 50,
        "num_episodes": 21,  # Number of complete self-play games to simulate during a new iteration.
        "max_episode_length": 150,
        "max_memory": 1000,
        "updateThreshold": 0.5,
        "discount_factor": 0.90,
        "batch_size": 32,
        "epsilon": 0.05,
        "temperature": 0.05,
        # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        "arenaCompare": 11,  # Number of games to play during arena play to determine if new net will be accepted.
        "checkpoint": "./NNEts/trained/",
        "load_folder_file": ("./NNets/trained/", "best.hdf5"),
    }
)


def main():
    g = TwoPlayerSnakeGame(board_x=20, board_y=20)
    nnet = NNetWrapper(g)
    c = Coach(g, nnet, args)
    c.learn()

    return 0


if __name__ == "__main__":
    main()
