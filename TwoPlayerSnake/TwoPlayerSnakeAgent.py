import numpy as np


class RandomPlayer:
    def __init__(self):
        pass
    def play(self, _):  # later this should have additional args that extend the board
        a = np.random.randint(4)
        return a