import sys

from TwoPlayerSnake.TwoPlayerSnakeGame import TwoPlayerSnakeGame
from TwoPlayerSnake.TwoPlayerSnakeArena import TwoPlayerSnakeArena
from TwoPlayerSnake.TwoPlayerSnakeAgent import RandomPlayer
from NNets.NNet import NNetWrapper as nn
from main import args
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="Output stats or a single live game",
        default="stats",
        choices=["stats", "game"],
    )
    return parser


def main():

    parser = get_parser()
    local_args = parser.parse_args()

    g = TwoPlayerSnakeGame(board_x=10, board_y=10)

    p1 = nn(g)
    p1.load_checkpoint(folder="./NNets/trained", filename="best.hdf5")
    p2 = RandomPlayer()
    arena = TwoPlayerSnakeArena(p1, p2, g, args)
    if local_args.mode == "game":
        arena.compare_two_models(display=True)
    else:
        arena.compare_two_models_n_times(20, verbose=True)


if __name__ == "__main__":
    main()
