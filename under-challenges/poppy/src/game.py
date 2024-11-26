
"""
game.py

This file contains the main game loop and logic for the game.
"""
from typing import Optional

ROWS = 6
COLUMNS = 7

EMPTY = 2
RED = 1
YELLOW = 0

DROP = 0
POP = 1

class Game:
    def __init__(self):
        self.state = [[EMPTY for _ in range(COLUMNS)] for _ in range(ROWS)]
        self.winner = None
        self.current_player = RED

    def switch_player(self):
        """
        Switch the current player.
        """
        self.current_player = YELLOW if self.current_player == RED else RED

    def is_running(self):
        """
        Check if the game is still running.
        The game is running if there is no winner yet.
        """
        return self.winner is None

    def print_board(self):
        """
        Print the current state of the board.
        """
        for row in self.state:
            print("|", end=" ")
            for cell in row:
                if cell == EMPTY:
                    print(" ", end=" ")
                elif cell == RED:
                    print("X", end=" ")
                elif cell == YELLOW:
                    print("O", end=" ")
                
                print("|", end=" ")
            
            print()

        print("  1   2   3   4   5   6   7")

    def is_valid_move(self, column, mode, player):
        """
        Check if a move is valid.
        """
        if mode == DROP:
            return self.state[0][column] == EMPTY
        elif mode == POP:
            return self.state[ROWS - 1][column] == player
    
    def make_move(self, column, mode, player):
        """
        Make a move on the board.
        """
        if mode == DROP:
            # Find the first empty cell in the column, starting from the bottom (row N_ROWS - 1)
            for row in range(ROWS - 1, -1, -1):
                if self.state[row][column] == EMPTY:
                    self.state[row][column] = player
                    break
        elif mode == POP:
            # Remove the bottom-most piece of the player's color in the column, and
            # shift all pieces above it down

            # Start from the bottom and copy the cell above it to the current cell.
            for row in range(ROWS - 1, 0, -1):
                self.state[row][column] = self.state[row - 1][column]

            # Set the top cell to EMPTY
            self.state[0][column] = EMPTY

        if (winner := self.check_for_win()) is not None:
            self.winner = winner

    def check_for_win(self) -> Optional[int]:
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for direction in directions:
            for row in range(ROWS):
                for column in range(COLUMNS):
                    if (ret := self.check_direction(row, column, direction)) is not None:
                        return ret

        return None

    def check_direction(self, row, column, direction) -> Optional[int]:
        player = self.state[row][column]
        if player == EMPTY:
            return None

        for i in range(1, 4):
            new_row = row + i * direction[0]
            new_column = column + i * direction[1]

            if new_row < 0 or new_row >= ROWS or new_column < 0 or new_column >= COLUMNS:
                return None

            if self.state[new_row][new_column] != player:
                return None
        
        return player

    def copy(self) -> 'Game':
        """
        Create a copy of the game state.
        """
        new_game = Game.__new__(Game)
        new_game.state = [row.copy() for row in self.state]
        new_game.winner = self.winner
        new_game.current_player = self.current_player

        return new_game
