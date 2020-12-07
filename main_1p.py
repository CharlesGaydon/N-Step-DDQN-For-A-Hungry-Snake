import argparse

from OnePlayerSnake.OnePlayerSnakeGame import OnePlayerSnakeGame
from OnePlayerSnake.OnePlayerSnakeCoach import Coach
from NNets.NNet import NNetWrapper

from utils import dotdict

args = dotdict(
    {
        "num_episodes": 10000,  #  number of episodes before stopping learning
        "min_reward_to_consider_episode": 1,
        "probability_to_keep_low_reward_experiences": 0.05,
        "max_episode_length": 150,
        "max_memory": 20000,  # number of experiences kept in memory
        "n_fit_update_target_nnet": 250,
        "printing_loss_frequency": 50,
        "discount_factor": 0.50,  # higher = long term interest
        "n_step_learning": 5,  # n for n-step sarsa
        "num_replay": 5,  # after each episode step, how many time do we train the model
        "batch_size": 64,
        "num_experience_to_start_learning": 100,
        "epsilon": 0.50,
        "min_epsilon": 0.1,
        "epsilon_decay_rate": 0.995,
        "arenaCompare": 20,  # Number of games to play during arena play to determine if new net will be accepted.
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
    g = OnePlayerSnakeGame(board_x=8, board_y=8)
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
