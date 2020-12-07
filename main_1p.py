import argparse

from OnePlayerSnake.OnePlayerSnakeGame import OnePlayerSnakeGame
from OnePlayerSnake.OnePlayerSnakeCoach import Coach
from NNets.NNet import NNetWrapper

from utils import dotdict

args = dotdict(
    {
        "num_episodes": 10000,  #  number of episodes before stopping learning
        "probability_to_keep_low_reward_experiences": 0.05,
        "frequency_to_control_interest_of_episode_every": 75,
        "min_reward_per_steps_to_consider_episode": 1.0 / 50,
        "max_memory": 20000,  # number of experiences kept in memory
        "num_experience_to_start_learning": 100,
        "discount_factor": 0.30,  # higher = long term interest
        "n_step_learning": 3,  # n for n-step sarsa
        "num_replay": 5,  # after each episode step, how many time do we train the model
        "batch_size": 64,
        "epsilon": 0.60,
        "min_epsilon": 0.1,
        "temperature": 0.1,  # lower for greedier decision and unstucking the snake
        "epsilon_decay_rate": 0.99,
        "arenaCompare": 20,  # Number of games to play during arena play to determine if new net will be accepted.
        "n_fit_update_target_nnet": 1000,
        "printing_loss_frequency": 50,
        "save_every_n_episodes": 50,  # save models with this frequency
        "load_folder_file": ("./NNets/OnePlayer/trained/", "best.hdf5"),
    }
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        help="Output stats or a single live game",
        default="False",
        choices=["True", "False"],
    )
    return parser


def main():
    g = OnePlayerSnakeGame(board_x=5, board_y=5)
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
