import time

import numpy as np
from tqdm import tqdm

# Using experience replay and Target Network as mentionned here:
# https://fr.slideshare.net/LeejinJeong/deep-sarsa-deep-qlearning-dqn-102870392


class OnePlayerSnakeArena:
    """
    Class that run an episode of the game and will build
    the list train_examples : a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                            pi is the MCTS informed policy vector, v is +1 if
                            the player eventually won the game, else -1.
    Player are object with predict method that takes board as argument and outputs a policy.
    """

    def __init__(self, nnet, game, args):
        self.nnet = nnet
        self.game = game
        self.args = args
        self.train_examples_memory = []

    # From Sutton & Barto 2018 - Reinforcement Learning Introduction Second Edition
    # From https://arxiv.org/abs/1710.02298 on combining Experience Replay and N-step DQLearning
    def deep_q_learning(self):

        for _ in tqdm(range(self.args.num_episodes), desc="Self Play and Learn"):

            self.game.init_game()
            n = self.args.n_step_learning
            S = [None] * (n + 1)
            R = [None] * (n + 1)
            A = [None] * (n + 1)

            S[0] = self.game.get_board()
            A[0] = self.nnet.epsilon_greedy_policy(
                self.game,
                perspective=None,
                args=self.args,
                forbidden_direction=self.game.p1_direction,
            )

            T = np.inf
            t = 0
            while True:
                if t < T:
                    # execute and get reward
                    R[(t + 1) % (n + 1)] = self.game.step(
                        a1=A[t % (n + 1)], display=False
                    )
                    S[(t + 1) % (n + 1)] = self.game.get_board()
                    game_ended = 1 * (self.game.status > 0)
                    if game_ended:
                        T = t + 1
                    else:
                        A[(t + 1) % (n + 1)] = self.nnet.epsilon_greedy_policy(
                            self.game,
                            perspective=None,
                            args=self.args,
                            forbidden_direction=self.game.p1_direction,
                        )
                # tau is the time whose estimate will be updated
                tau = t - n + 1
                if tau >= 0:
                    G = 0  # expected return from state at time tau
                    for i in range(tau + 1, min(tau + n, T) + 1):
                        G += (self.args.discount_factor ** (i - tau + 1)) * R[
                            i % (n + 1)
                        ]
                    if tau + n < T:
                        # a prediction will have to be made in addition to G
                        # based on the state at tau + n
                        game_ended = False
                        experience = (
                            S[tau % (n + 1)],
                            A[tau % (n + 1)],
                            G,
                            game_ended,
                            S[(tau + n) % (n + 1)],
                        )
                    else:
                        # no prediction is needed as we are at the end of the episode
                        game_ended = True
                        experience = (
                            S[tau % (n + 1)],
                            A[tau % (n + 1)],
                            G,
                            game_ended,
                            None,
                        )
                    self.train_examples_memory.append(experience)

                if tau == T - 1:
                    # We updated all experiences, and training for this episode is over
                    break

                # Stop episode if it is too long
                if self.game.episode_duration > self.args.max_episode_length:
                    print(
                        f"Episode lasted more that max_episode_length: "
                        f"{self.game.episode_duration}/{self.args.max_episode_length}"
                    )
                    break

                # next time index
                t = t + 1

            # Fit the model now that we have at least batch_size experiences
            enough_memory = (
                len(self.train_examples_memory)
                > self.args.num_experience_to_start_learning
            )

            if enough_memory:

                for replay in range(self.args.num_replay):

                    self.nnet.optimize_network(self.train_examples_memory, self.args)
                    if self.nnet.fit_is_called_counter % 1000 == 0:
                        print(
                            f"Fit was called {self.nnet.fit_is_called_counter} times since training started."
                        )

                    # Update the target NNet every n_fit_update_target_nnet steps
                    if (
                        self.nnet.fit_is_called_counter
                        % self.args.n_fit_update_target_nnet
                        == 0
                    ):
                        self.nnet.update_target_nnet()
                        print(f" Updating target nnet.")

                    if (
                        self.nnet.fit_is_called_counter
                        % self.args.printing_loss_frequency
                        == 0
                    ):
                        print(
                            f" Mean loss = {np.mean(self.nnet.loss_historic).round(3)} (N={len(self.nnet.loss_historic)})"
                        )

            # Cut Memory
            if len(self.train_examples_memory) > self.args.max_memory:
                self.train_examples_memory = self.train_examples_memory[
                    -self.args.max_memory :
                ]
            # Decay the exploration parameter
            self.args.epsilon = max(
                self.args.min_epsilon, self.args.epsilon_decay_rate * self.args.epsilon
            )

    def play_one_game(self, display=False):

        self.game.init_game()

        while not self.game.status:
            # we predict directly using the NN
            if display:
                print(
                    self.nnet.predict_action_values_from_game(self.game, perspective=1)
                )
            action_p1 = self.nnet.greedy_policy(self.game, perspective=1)

            self.game.step(a1=action_p1, display=display)

            if display:
                self.game.display()
                time.sleep(0.15)
            if self.game.episode_duration > self.args.max_episode_length:
                print(
                    f" Evaluation episode lasted more that max_episode_length: "
                    f"{self.game.episode_duration}/{self.args.max_episode_length}"
                )
                break

    def play_n_games(self, nb_games, verbose=True):

        stats = []
        if nb_games != 0:
            for _ in tqdm(range(nb_games), desc="Evaluating model"):
                self.play_one_game(display=False)
                stats.append(self.game.get_game_stats())
            if verbose:
                m1, m2 = (
                    np.mean([s[0] for s in stats]).round(2),
                    np.mean([s[1] for s in stats]).round(2),
                )
                # print("\n".join([f"{s[0]} vs {s[1]}" for s in stats]))
                print(f"Means of episode duration & total reward: {m1} - {m2}")

        return stats
