from TwoPlayerSnake.TwoPlayerSnakeGame import TwoPlayerSnakeGame
from TwoPlayerSnake.TwoPlayerSnakeArena import TwoPlayerSnakeArena
from TwoPlayerSnake.TwoPlayerSnakeAgent import RandomPlayer

p1 = RandomPlayer()
p2 = RandomPlayer()
g = TwoPlayerSnakeGame(n=4, m=4)

arena = TwoPlayerSnakeArena(p1, p2, g)
arena.play_game()
