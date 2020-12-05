import time

import numpy as np
from tqdm import tqdm

# Using experience replay and Target Network as mentionned here:
# https://fr.slideshare.net/LeejinJeong/deep-sarsa-deep-qlearning-dqn-102870392


class TwoPlayerSnakeArena:
    """
    Class that run an episode of the game and will build
    the list train_examples : a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                            pi is the MCTS informed policy vector, v is +1 if
                            the player eventually won the game, else -1.
    Player are object with predict method that takes board as argument and outputs a policy.
    """

    def __init__(self, nnet, player_2, game, args):
        self.nnet = nnet
        self.player_2 = player_2
        self.game = game
        self.train_examples_memory = []
        self.args = args

    def deep_q_learning(self):

        for _ in tqdm(range(self.args.num_episodes), desc="Self Play and Learn"):

            self.game.init_game()
            # store init state and store s
            last_s1 = self.game.get_board(1)
            last_s2 = self.game.get_board(2)

            while not self.game.status:
                # select next action for both player
                action_p1 = self.nnet.epsilon_greedy_policy(
                    self.game, self.args, perspective=1
                )
                action_p2 = self.player_2.epsilon_greedy_policy(
                    self.game, self.args, perspective=2
                )
                # execute and get reward for both snakes
                r1, r2 = self.game.step(a1=action_p1, a2=action_p2, display=False)
                s1 = self.game.get_board(1)
                s2 = self.game.get_board(2)
                game_ended = 1 * (self.game.status > 0)

                ### TODO: modify this to implement n-step q-learning

                sarsa1 = (last_s1, action_p1, r1, game_ended, s1)
                sarsa2 = (last_s2, action_p2, r2, game_ended, s2)
                self.train_examples_memory.extend([sarsa1, sarsa2])

                if len(self.train_examples_memory) > self.args.batch_size:
                    # start learning when sufficient examples
                    self.nnet.optimize_network(self.train_examples_memory, self.args)

                # update last state
                last_s1 = s1
                last_s2 = s2
                if self.game.episode_duration > self.args.max_episode_length:
                    print(
                        f"Episode lasted more that max_episode_length: "
                        f"{self.game.episode_duration}/{self.args.max_episode_length}"
                    )
                    break
            # update after each episode
            self.nnet.update_target_nnet()
            # Print the loss
            print(
                f" Average loss: {np.mean(self.nnet.loss_historic)} (N={len(self.nnet.loss_historic)})"
            )

            if len(self.train_examples_memory) > self.args.max_memory:
                self.train_examples_memory = self.train_examples_memory[
                    -self.args.max_memory :
                ]

    def compare_two_models(self, display=False):

        self.game.init_game()

        while not self.game.status:
            # we predict directly using the NN
            action_p1 = self.nnet.greedy_policy(self.game, self.args, perspective=1)
            action_p2 = self.player_2.greedy_policy(self.game, self.args, perspective=2)

            self.game.step(a1=action_p1, a2=action_p2, display=display)

            if display:
                self.game.display()
                time.sleep(0.1)

    def compare_two_models_n_times(self, nb_games, verbose=True):
        wins = 0
        draws = 0
        loss = 0
        stats = []
        if nb_games != 0:
            for _ in tqdm(range(nb_games), desc="Compare models"):
                self.compare_two_models(display=False)
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
                # print("\n".join([f"{s[0]} vs {s[1]}" for s in stats]))
                print(f"Average lenghts of p1 vs. p2: {m1} - {m2}")
                print(
                    f"Results wins|draws|loss: {wins}|{draws}|{loss} vs other player.\n"
                )
        return wins, draws, loss, stats
