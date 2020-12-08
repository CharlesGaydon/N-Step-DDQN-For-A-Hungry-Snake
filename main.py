import argparse

from Snake_DDQN.game import Game
from Snake_DDQN.coach import Coach
from Snake_DDQN.nnet_wrapper import NNetWrapper

from utils import dotdict

args = dotdict(
    {
        "num_episodes": 10000,  # number of episodes before stopping learning
        "frequency_to_control_interest_of_episode_every": 75,  # check if episode has enough reward to be of interest
        "min_reward_per_steps_to_consider_episode": 1.0
        / 50,  # threshold of an "interesting" episode
        "probability_to_keep_low_reward_experiences": 0.05,  # proba to keep "non-interesting" episode
        "max_memory": 20000,  # number of experiences kept in memory
        "num_experience_to_start_learning": 100,  # start learning after x nb of (S,A,R,S,A) experiences in memory
        "discount_factor": 0.30,  # N.B.: higher = long term interest
        "n_step_learning": 3,  # Times in future to consider for calculation of return in n-step Q-Learning.
        "num_replay": 5,  # after each episode ended, number of times we fit on a minibatch
        "batch_size": 64,  # Minibatch size
        "epsilon": 0.60,  # Start exploration parameter for the epsilon-policy (softmax with temperature)
        "epsilon_decay_rate": 0.99,  # How much we decay the exploration after each episode
        "min_epsilon": 0.1,  # Min epsilon reached after decay
        "temperature": 0.1,  # For softmax: lower for greedier decision.
        "arenaCompare": 20,  # Number of games to play during when we evaluate the agent.
        "n_fit_update_target_nnet": 1000,  # Frequency of update of the target network in Double Deep Q-Learning
        "printing_loss_frequency": 50,  # Frequency to print running average of loss and epsilon value.
        "save_every_n_episodes": 50,  # Frequency to save models' weights
        "load_folder_file": (
            "./Trained_Models/",
            "final_best.hdf5",
        ),  # path to store/load the weights.
    }
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        help="Whether to load saved weight or to start learning from scratch",
        default="False",
        choices=["True", "False"],
    )
    return parser


def main():
    g = Game(board_x=5, board_y=5)
    parser = get_parser()
    local_args = parser.parse_args()
    if local_args.resume == "True":
        nnet = NNetWrapper(g, load_folder_file=args.load_folder_file)
    else:
        nnet = NNetWrapper(g)
    c = Coach(g, nnet, args)
    c.learn()

    return 0


if __name__ == "__main__":
    main()
