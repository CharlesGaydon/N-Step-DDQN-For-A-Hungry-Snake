from TwoPlayerSnake.TwoPlayerSnakeGame import TwoPlayerSnakeGame
from coach import Coach
from NNets.NNet import NNetWrapper

from utils import dotdict

args = dotdict(
    {
        "numIters": 30,
        "num_episodes": 20,  # Number of complete self-play games to simulate during a new iteration.
        "max_episode_length": 200,
        "updateThreshold": 0.5,  # always accept
        "batch_size": 64,
        "epsilon": 0.05,
        "discount_factor": 0.35,
        "temperature": 0.5,
        # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        "arenaCompare": 11,  # Number of games to play during arena play to determine if new net will be accepted.
        "checkpoint": "./NNEts/trained/",
        "load_folder_file": ("./NNets/trained/", "best_checkpoint"),
    }
)


def main():
    g = TwoPlayerSnakeGame(board_x=8, board_y=8)
    nnet = NNetWrapper(g)
    c = Coach(g, nnet, args)
    c.learn()

    return 0


if __name__ == "__main__":
    main()
