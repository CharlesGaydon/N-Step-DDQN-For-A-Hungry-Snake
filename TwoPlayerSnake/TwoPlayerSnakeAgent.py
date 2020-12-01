import numpy as np


class RandomPlayer:
    def __init__(self):
        self.pi = None
        self.action = None
        pass

    def predict(self, _):
        # later this should have additional args that extend the board
        self.pi = np.random.random(4)
        return self.pi, None
