import sys

from OnePlayerSnake.OnePlayerSnakeGame import OnePlayerSnakeGame
from OnePlayerSnake.OnePlayerSnakeArena import OnePlayerSnakeArena
from NNets.NNet import NNetWrapper as nn
from main_1p import args
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

    g = OnePlayerSnakeGame(board_x=5, board_y=5)

    p1 = nn(g, load_folder_file=args.load_folder_file)

    arena = OnePlayerSnakeArena(p1, g, args)
    if local_args.mode == "game":
        arena.play_one_game(display=True)
    else:
        arena.play_n_games(30, verbose=True)


if __name__ == "__main__":
    main()
