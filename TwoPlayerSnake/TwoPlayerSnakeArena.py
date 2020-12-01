import time


class TwoPlayerSnakeArena:
    def __init__(self, p1, p2, game):
        self.player1 = p1
        self.player2 = p2
        self.game = game

    def play_game(self):

        while not self.game.status:

            action1 = self.player1.play(self.game.get_board(1))
            action2 = self.player2.play(self.game.get_board(2))

            self.game.step(action1=action1, action2=action2)
            self.game.display()
            time.sleep(0.2)
