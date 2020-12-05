import sys

import numpy as np

direction_encoding = {
    0: (-1, 0),  # "up"
    1: (0, 1),  # "right"
    2: (1, 0),  # "bottom"
    3: (0, -1),  # "left"
}


class TwoPlayerSnakeGame:
    def __init__(self, board_x=20, board_y=30):
        self.board_x = board_x
        self.board_y = board_y
        self.action_size = 4
        self.board = None  # shape will be (n, m)
        self.status = None
        self.episode_duration = 0
        self.init_game()

    def init_game(self):
        # set player 1 initial position and direction
        self.p1_direction = 1
        self.p1_positions = [
            (self.board_x // 5, self.board_y // 5),
            (self.board_x // 5, self.board_y // 5 + 1),
        ]
        self.p1_ate_apple = False
        # set player 2 initial position
        self.p2_positions = [
            (4 * self.board_x // 5, 4 * self.board_y // 5),
            (4 * self.board_x // 5, 4 * self.board_y // 5 - 1),
        ]
        self.p2_direction = 3
        self.p2_ate_apple = False

        # init the state from its precursors
        self.set_apple_position()
        self.set_board_from_positions()

        # 0: ongoing, 1: p1 wins, 2: p2 wins, 3: both players lose and it's a draw
        self.status = 0
        self.episode_duration = 0

    def step(self, a1, a2, display=False):
        self.episode_duration += 1

        # if the new action is the opposite of the previous one, it means we keep the previous one
        if not (a1 + 2) % 4 == self.p1_direction:
            self.p1_direction = a1
        if not (a2 + 2) % 4 == self.p2_direction:
            self.p2_direction = a2

        # update the positions
        self.p1_positions = self.extend_snake(
            self.p1_positions, self.p1_direction, self.p1_ate_apple
        )
        self.p2_positions = self.extend_snake(
            self.p2_positions, self.p2_direction, self.p2_ate_apple
        )
        p1_crashed = p2_crashed = False
        # test if the last move lead to a position that collapses with each other = crash
        if (self.p1_positions[-1] in self.p1_positions[:-1] + self.p2_positions) or (
            self.is_out_of_board(self.p1_positions[-1])
        ):
            p1_crashed = True
        if (self.p2_positions[-1] in self.p1_positions + self.p2_positions[:-1]) or (
            self.is_out_of_board(self.p2_positions[-1])
        ):
            p2_crashed = True

        # status is the winning player, or 3 if both lost
        if p1_crashed and p2_crashed:
            self.status = 3
        elif p1_crashed:
            self.status = 2
        elif p2_crashed:
            self.status = 1

        if any((p1_crashed, p2_crashed)):
            if display:
                print("Final state:")
                print("A:", self.p1_positions, self.p1_direction)
                print("B", self.p2_positions, self.p2_direction)
                print(f"Final game status : {self.status}")
            # self.set_board_from_positions()
            return (-35 * p1_crashed, -35 * p2_crashed)

        # test if the apple was caught and update in consequence
        if self.p1_positions[-1] == self.apple_position:
            self.p1_ate_apple = True
            self.p2_ate_apple = False
            self.set_apple_position()
        elif self.p2_positions[-1] == self.apple_position:
            self.p1_ate_apple = False
            self.p2_ate_apple = True
            self.set_apple_position()
        else:
            self.p1_ate_apple = False
            self.p2_ate_apple = False

        # status is 0 and  the game can continue:
        self.set_board_from_positions()
        self.status = 0

        # return a reward if an apple is eaten
        return -0.25 + self.p1_ate_apple * 5, -0.25 + self.p2_ate_apple * 5

    def is_out_of_board(self, position):
        try:
            self.board[position]
            return False
        except IndexError:
            return True

    def set_board_from_positions(self):
        self.board = np.zeros((self.board_x, self.board_y))
        for p in self.p1_positions[:-1]:
            self.board[p] = 1
        for p in self.p2_positions[:-1]:
            self.board[p] = -1
        self.board[self.p1_positions[-1]] = 2
        self.board[self.p2_positions[-1]] = -2

        self.board[self.apple_position] = 10
        # special case to display crashing snake - could be done separately
        if self.p1_positions[-1] in self.p1_positions[:-1] + self.p2_positions:
            self.board[self.p1_positions[-1]] = 3
        if self.p2_positions[-1] in self.p1_positions + self.p2_positions[:-1]:
            self.board[self.p2_positions[-1]] = 3

        return self.board

    def set_apple_position(self):
        # TODO: replace with a random selection among the complementary set of the positions
        while True:
            self.apple_position = (
                np.random.randint(self.board_x),
                np.random.randint(self.board_y),
            )
            if (self.apple_position not in self.p1_positions) and (
                self.apple_position not in self.p2_positions
            ):
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

    def get_board(self, player_id):
        if player_id == 1:
            return self.board.copy()
        elif player_id == 2:
            # reverse the board encoding to get perspective of player 2
            reverse_board = self.board.copy()
            reverse_board[
                (1 <= (reverse_board * reverse_board))
                & ((reverse_board * reverse_board) <= 4)
            ] = -reverse_board[
                (1 <= (reverse_board * reverse_board))
                & ((reverse_board * reverse_board) <= 4)
            ]
            return reverse_board
        else:
            raise KeyError(f"Unknown player_id: {player_id}")

    def get_board_dimensions(self):
        return self.board_x, self.board_y

    def get_game_stats(self):
        stats = [len(self.p1_positions), len(self.p2_positions)]
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
