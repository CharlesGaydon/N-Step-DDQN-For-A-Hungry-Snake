import time

import numpy as np
from tqdm import tqdm


class TwoPlayerSnakeArena:
    """
    Class that run an episode of the game and will build
    the list train_examples : a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                            pi is the MCTS informed policy vector, v is +1 if
                            the player eventually won the game, else -1.
    Player are object with predict method that takes board as argument and outputs a policy.
    """

    def __init__(self, p1, p2, game):
        self.player1 = p1
        self.player2 = p2
        self.game = game
        self.train_examples = []

    def play_game(self, training_mode=True, display=False):

        self.game.init_game()
        self.train_examples = []

        while not self.game.status:
            if not training_mode:
                # we predict directly using the NN
                pi_p1, _ = self.player1.predict(self.game, perspective=1)
                pi_p2, _ = self.player2.predict(self.game, perspective=2)
            else:
                # we use the v values
                pi_p1, _ = self.player1.make_policy_from_q_values(
                    self.game, perspective=1
                )
                pi_p2, _ = self.player2.make_policy_from_q_values(
                    self.game, perspective=2
                )
                # we remember the decisions we took
                self.train_examples.append([self.game.get_board(1), 1, pi_p1, None])
                self.train_examples.append([self.game.get_board(2), 2, pi_p2, None])

            # make the decision and update the game
            action_p1 = np.random.choice(4, p=pi_p1 / pi_p1.sum())
            action_p2 = np.random.choice(4, p=pi_p2 / pi_p2.sum())

            self.game.step(action1=action_p1, action2=action_p2, display=display)

            if display:
                self.game.display()
                time.sleep(0.1)

        if training_mode:
            if self.game.status == 3:
                # TODO: try ignoring draws to have more balancer examples !
                def scorer(_):
                    return 0

            else:

                def scorer(player_id):
                    return (-1) ** (player_id != self.game.status)

            self.train_examples = [
                [x[0], x[2], scorer(x[1])] for x in self.train_examples
            ]

    def compare_two_models(self, nb_games, verbose=True):
        wins = 0
        draws = 0
        loss = 0
        stats = []
        for comparison_play in tqdm(range(nb_games), desc="Compare models"):
            self.play_game(training_mode=False, display=False)
            r = self.game.status
            if r == 1:
                wins += 1
            elif r == 2:
                loss += 1
            elif r == 3:
                draws += 1
            stats.append(self.game.get_game_stats())
        if verbose:
            m1, m2 = (
                np.mean([s[0] for s in stats]).round(2),
                np.mean([s[1] for s in stats]).round(2),
            )
            print("\n".join([f"{s[0]} vs {s[1]}" for s in stats]))
            print(f"Average lenghts of p1 vs. p2: {m1} - {m2}")
            print(f"Results wins|draws|loss: {wins}|{draws}|{loss} vs other player.\n")
        return wins, draws, loss, stats
