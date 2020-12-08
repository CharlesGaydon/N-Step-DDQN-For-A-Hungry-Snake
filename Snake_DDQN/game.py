import sys
import numpy as np

direction_encoding = {
    0: (-1, 0),  # "up"
    1: (0, 1),  # "right"
    2: (1, 0),  # "bottom"
    3: (0, -1),  # "left"
}
APPLE_REWARD = 10
CRASH_REWARD = -20
CLOSENESS_TO_APPLE_REWARD = (
    2  # times fraction of a pseudo-max possible distance to the apple
)


class Game:
    def __init__(self, board_x=8, board_y=8):
        self.board_x = board_x + 2  # borders
        self.board_y = board_y + 2  # borders
        self.action_size = 4
        self.board = None  # shape will be (n, m)
        self.status = 0
        self.p1_ate_apple = False
        self.episode_duration = 0
        self.total_reward = 0

        self.init_game()

    def init_game(self):
        # 0: ongoing, 1: crashed
        self.status = 0
        self.episode_duration = 0
        self.total_reward = 0

        # set player 1 initial position and direction
        self.p1_direction = np.random.randint(4)
        dx, dy = direction_encoding[self.p1_direction]
        self.p1_positions = [
            (self.board_x // 2, self.board_y // 2),
            (self.board_x // 2 + dx, self.board_y // 2 + dy),
        ]
        self.p1_ate_apple = False

        # init the state from its precursors
        self.set_apple_position()
        self.set_board_from_positions()

        self.current_experience = []

    def step(self, a1, display=False):
        self.episode_duration += 1

        # if the new action is the opposite of the previous one, it means we keep the previous one
        if not (a1 + 2) % 4 == self.p1_direction:
            self.p1_direction = a1

        # update the positions
        self.p1_positions = self.extend_snake(
            self.p1_positions, self.p1_direction, self.p1_ate_apple
        )

        # test if snake crashed
        p1_crashed = self.is_out_of_board(self.p1_positions[-1])
        if p1_crashed:
            self.status = 1
            if display:
                print("Final state:")
                print("Position:", self.p1_positions)
                print(
                    "Direction:",
                    self.p1_direction,
                    direction_encoding[self.p1_direction],
                )
                print(f"Final game status : {self.status}")
            return CRASH_REWARD

        # test if the apple was caught and update in consequence
        if self.p1_positions[-1] == self.apple_position:
            self.total_reward += 1
            self.p1_ate_apple = True
            self.set_apple_position()
        else:
            self.p1_ate_apple = False

        # the game can continue:
        self.set_board_from_positions()
        self.status = 0

        reward = self.p1_ate_apple * APPLE_REWARD

        # add reward if getting closer to the apple on x axis or y axis
        if not self.p1_ate_apple:
            reward += self.get_closeness_to_apple() * CLOSENESS_TO_APPLE_REWARD

        return reward

    def is_out_of_board(self, position):
        # returns True if snake head touches the border of the board
        if position[0] < 1 or self.board_x - 1 <= position[0]:
            return True
        if position[1] < 1 or self.board_y - 1 <= position[1]:
            return True
        return False

    def get_closeness_to_apple(self):
        # Returns euclidian distance from snake head to apple
        x, y = self.p1_positions[-1]
        a_x, a_y = self.apple_position
        dist = np.sqrt(np.power(a_x - x, 2) + np.power(a_y - y, 2))
        max_dist = np.sqrt((self.board_x - 2) ** 2 + (self.board_y - 2) ** 2)
        return max_dist - dist

    def set_board_from_positions(self):
        self.board = np.zeros((self.board_x, self.board_y))

        # borders
        self.board[0, :] = 3
        self.board[-1, :] = 3
        self.board[:, 0] = 3
        self.board[:, -1] = 3

        # snakes body and head
        for p in self.p1_positions[:-1]:
            self.board[p] = 1

        self.board[self.p1_positions[-1]] = 2

        # apple
        self.board[self.apple_position] = 10

        return self.board

    def set_apple_position(self):
        while True:
            self.apple_position = (
                np.random.randint(1, self.board_x - 1),
                np.random.randint(1, self.board_y - 1),
            )
            if self.apple_position not in self.p1_positions:
                break

    def extend_snake(self, positions, direction, ate_apple):
        # inplace operation

        current_position_x, current_position_y = positions[-1]
        dx, dy = direction_encoding[direction]
        next_position = (current_position_x + dx, current_position_y + dy)
        # extend snake
        positions.append(next_position)

        # move the tail if no apple was eaten
        if not ate_apple:
            del positions[0]
        return positions

    def add_to_current_experience(self, experience):
        # experience: sarsa like tuple
        self.current_experience.append(experience)

    def get_board(self, perspective=None):
        scaled_board = self.board.copy()
        scaled_board = scaled_board - scaled_board.min()
        scaled_board = scaled_board / scaled_board.max()
        return scaled_board

    def get_board_dimensions(self):
        return self.board_x, self.board_y

    def get_game_stats(self):
        stats = [self.episode_duration, self.total_reward]
        return stats

    def get_reward_per_episode_steps(self):
        return 1.0 * self.total_reward / self.episode_duration

    def display(self):
        output_str = " " + "___" * self.board_y + "\n"
        for x in range(self.board_x):
            output_str += "|"
            for y in range(self.board_y):
                if self.board[x, y] == 0:
                    output_str += " . "
                elif self.board[x, y] == 10:
                    output_str += " O "
                elif self.board[x, y] in [1, 2]:
                    output_str += " S "
                elif self.board[x, y] in [-1, -2]:
                    output_str += " B "
                elif self.board[x, y] == 3:
                    output_str += " X "
                else:
                    raise ValueError(f"wrong encoding in display {self.board[x,y]}")
            output_str += "|\n"
        output_str += " "

        output_str += "———" * self.board_y + "\n\n \r"
        sys.stdout.write(output_str)
