from TwoPlayerSnake.TwoPlayerSnakeGame import TwoPlayerSnakeGame
from TwoPlayerSnake.TwoPlayerSnakeArena import TwoPlayerSnakeArena
from TwoPlayerSnake.TwoPlayerSnakeAgent import RandomPlayer
from model.NNet import NNetWrapper as nn

g = TwoPlayerSnakeGame(board_x=4, board_y=4)

p1 = nn(g)
p2 = RandomPlayer()

arena = TwoPlayerSnakeArena(p1, p2, g)
arena.play_game(mode="display")
