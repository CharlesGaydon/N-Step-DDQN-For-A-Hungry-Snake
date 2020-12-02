from TwoPlayerSnake.TwoPlayerSnakeGame import TwoPlayerSnakeGame
from TwoPlayerSnake.TwoPlayerSnakeArena import TwoPlayerSnakeArena
from TwoPlayerSnake.TwoPlayerSnakeAgent import RandomPlayer
from NNets.NNet import NNetWrapper as nn

g = TwoPlayerSnakeGame(board_x=10, board_y=10)

p1 = nn(g)
p1.load_checkpoint(folder="./NNets/trained", filename="best.hdf5")
p2 = RandomPlayer()

arena = TwoPlayerSnakeArena(p1, p2, g)
arena.play_game(keep_track_of_historic=False, display=True)
