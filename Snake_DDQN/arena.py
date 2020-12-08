import time
import numpy as np
from tqdm import tqdm


class Arena:
    def __init__(self, nnet, game, args):
        self.nnet = nnet
        self.game = game
        self.args = args
        self.train_examples_memory = []

    # From Sutton & Barto 2018 - Reinforcement Learning Introduction Second Edition
    # From https://arxiv.org/abs/1710.02298 on combining Experience Replay and N-step DQLearning
    def n_steps_deep_q_learn(self):

        self.game.init_game()
        n = self.args.n_step_learning
        S = [None] * (n + 1)
        R = [None] * (n + 1)
        A = [None] * (n + 1)

        S[0] = self.game.get_board()
        A[0] = self.nnet.epsilon_hot_softmax_policy(
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
                R[(t + 1) % (n + 1)] = self.game.step(a1=A[t % (n + 1)], display=False)
                S[(t + 1) % (n + 1)] = self.game.get_board()
                game_ended = 1 * (self.game.status > 0)
                if game_ended:
                    T = t + 1
                else:
                    A[(t + 1) % (n + 1)] = self.nnet.epsilon_hot_softmax_policy(
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
                    G += (self.args.discount_factor ** (i - tau + 1)) * R[i % (n + 1)]
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
                self.game.add_to_current_experience(experience)

            if tau == T - 1:
                # We updated all experiences, and training for this episode is over
                break

            # Stop episode if it is too long without reward
            if (
                self.game.episode_duration
                % self.args.frequency_to_control_interest_of_episode_every
                == 0
            ):
                if (
                    self.game.get_reward_per_episode_steps()
                    < self.args.min_reward_per_steps_to_consider_episode
                ):
                    print(
                        f"Episode had low reward: {self.game.total_reward} for {self.game.episode_duration} steps"
                    )
                    if t < T:
                        T = t + n + 1
                        t = T  # add the last step to learning

            # next time index
            t = t + 1

        # Decide if the experience was interested enough to be used
        add_experience_to_memory = True
        if (
            self.game.get_reward_per_episode_steps()
            < self.args.min_reward_per_steps_to_consider_episode
        ):
            if not (
                np.random.random()
                < self.args.probability_to_keep_low_reward_experiences
            ):
                add_experience_to_memory = False

        # TODO LOOG
        if add_experience_to_memory:
            print(f"Using experience with reward {self.game.total_reward}")

        if add_experience_to_memory:
            self.train_examples_memory.extend(self.game.current_experience)

        # Fit the model if we have at least num_experience_to_start_learning experiences in memory
        enough_memory = (
            len(self.train_examples_memory) > self.args.num_experience_to_start_learning
        )
        if enough_memory:
            for replay in range(self.args.num_replay):
                # Fit the network num_replay times
                self.nnet.optimize_network(self.train_examples_memory, self.args)

                # Print every fit_is_called_counter fitting operation
                if self.nnet.fit_is_called_counter % 1000 == 0:
                    print(
                        f"Fit was called {self.nnet.fit_is_called_counter} times since training started."
                    )

                # ¨Print mean loss and also print out the epsilon value
                if (
                    self.nnet.fit_is_called_counter % self.args.printing_loss_frequency
                    == 0
                ):
                    print(
                        f" Mean loss = {np.mean(self.nnet.loss_historic).round(3)} (N={len(self.nnet.loss_historic)})"
                    )
                    print(f" Epsilon value = {self.args.epsilon}")

                # Update the target NNet every n_fit_update_target_nnet steps
                if (
                    self.nnet.fit_is_called_counter % self.args.n_fit_update_target_nnet
                    == 0
                ):
                    self.nnet.update_target_nnet()
                    print(f" Updating target nnet.")

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
            action_p1 = self.nnet.hot_softmax_policy(
                self.game,
                self.args,
                perspective=1,
                forbidden_direction=self.game.p1_direction,
            )

            self.game.step(a1=action_p1, display=display)

            if display:
                self.game.display()
                time.sleep(0.15)
            # if check step is reached
            if (
                self.game.episode_duration
                % self.args.frequency_to_control_interest_of_episode_every
                == 0
            ):
                # if not enough reward
                if (
                    self.game.get_reward_per_episode_steps()
                    < self.args.min_reward_per_steps_to_consider_episode
                ):
                    print(
                        f"Episode had low reward: {self.game.total_reward} for {self.game.episode_duration} steps"
                    )
                break

    def play_n_games(self, nb_games, verbose=True):

        stats = []
        if nb_games != 0:
            for _ in tqdm(range(nb_games), desc="Evaluating model"):
                self.play_one_game(display=False)
                stats.append(self.game.get_game_stats())
            if verbose:
                m1, m2, high = (
                    np.median([s[0] for s in stats]).round(2),
                    np.median([s[1] for s in stats]).round(2),
                    np.max([s[1] for s in stats]).round(2),
                )

                # print("\n".join([f"{s[0]} vs {s[1]}" for s in stats]))
                print(f"Median episode duration & median total reward: {m1} - {m2}")
                print(f"High score: {high}")

        return stats
