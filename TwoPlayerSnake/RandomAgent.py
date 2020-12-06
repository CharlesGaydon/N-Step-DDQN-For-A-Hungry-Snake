import numpy as np


class RandomPlayer:
    def __init__(self):
        self.pi = None
        self.action = None
        pass

    def greedy_policy(self, a_, b_, perspective=None):
        # later this should have additional args that extend the board
        self.action = np.random.choice(4)
        return self.action
