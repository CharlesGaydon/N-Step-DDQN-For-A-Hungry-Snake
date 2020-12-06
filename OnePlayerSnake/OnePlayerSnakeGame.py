import sys

import numpy as np

direction_encoding = {
    0: (-1, 0),  # "up"
    1: (0, 1),  # "right"
    2: (1, 0),  # "bottom"
    3: (0, -1),  # "left"
}
APPLE_REWARD = 1
CRASH_REWARD = -10
STEP_REWARD = -0.025


class OnePlayerSnakeGame:
    def __init__(self, board_x=20, board_y=30):
        self.board_x = board_x + 2  # borders
        self.board_y = board_y + 2  # borders
        self.action_size = 4
        self.board = None  # shape will be (n, m)
        self.status = None
        self.episode_duration = 0
        self.init_game()

    def init_game(self):
        # 0: ongoing, 1: crashed
        self.status = 0
        self.episode_duration = 0
        self.total_reward = 0

        # set player 1 initial position and direction
        self.p1_direction = 1
        self.p1_positions = [
            (self.board_x // 5, self.board_y // 5),
            (self.board_x // 5, self.board_y // 5 + 1),
        ]
        self.p1_ate_apple = False

        # init the state from its precursors
        self.set_apple_position()
        self.set_board_from_positions()

    def step(self, a1, display=False):
        self.episode_duration += 1

        # if the new action is the opposite of the previous one, it means we keep the previous one
        if not (a1 + 2) % 4 == self.p1_direction:
            self.p1_direction = a1

        # update the positions
        self.p1_positions = self.extend_snake(
            self.p1_positions, self.p1_direction, self.p1_ate_apple
        )

        # test if snake went out of board
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
            self.p1_ate_apple = True
            self.total_reward += APPLE_REWARD
            self.set_apple_position()
        else:
            self.p1_ate_apple = False
        # status is 0 and  the game can continue:
        self.set_board_from_positions()
        self.status = 0

        return -STEP_REWARD + self.p1_ate_apple * APPLE_REWARD

    def is_out_of_board(self, position):
        if position[0] < 1 or self.board_x - 1 <= position[0]:
            return True
        if position[1] < 1 or self.board_y - 1 <= position[1]:
            return True
        return False

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
        # Note: input is modified by this, which is expected for now

        current_position_x, current_position_y = positions[-1]
        move_x, move_y = direction_encoding[direction]
        next_position = (current_position_x + move_x, current_position_y + move_y)
        # extend snake
        positions.append(next_position)

        # move the tail if no apple was eaten
        if not ate_apple:
            del positions[0]
        return positions

    def get_board(self, perspective=None):
        return self.board.copy()

    def get_board_dimensions(self):
        return self.board_x, self.board_y

    def get_game_stats(self):
        stats = [self.episode_duration, self.total_reward]
        return stats

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
                    output_str += " A "
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
