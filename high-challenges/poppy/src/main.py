#!/usr/bin/env python3

import os
import mcts
import sys
import time
import math

from game import Game, DROP, POP, RED
from bot import State

READ_TIMEOUT_SECONDS = 2

def main():
    flag = 'ictf{k33p_1t_uP_4nd_1t_w1ll_b3_0k4y}'

    searcher = mcts.mcts(
        timeLimit=1_000, # 1 second
    )

    print("Welcome to Poppy!")
    print("You are playing against a bot. Try to get 4 in a row to win!")
    print("You can drop a piece in the top of a column or, if the bottom")
    print("piece in a column is yours, you can pop it out.")
    print("You must win two of three games to win the match and get the flag.")
    print()

    choice = input("Play in practice mode? [y/N]").strip().lower()
    is_practice_mode = choice == "y"
    if is_practice_mode:
        print("You are playing in practice mode. You can take as long as you want to make your move.")
        flag = "[REDACTED]"
    else:
        print("You are playing in challenge mode. You have 2 seconds to make your move.")

    n_human_wins = 0
    for n_games_played in range(3):
        game = Game()
        print(f"Game {n_games_played + 1} of 3")

        while game.is_running():
            print(f'Board:')
            game.print_board()

            print(f"Player {game.current_player}, it's your turn!")
            if game.current_player == RED:
                # # Human player
                t_start = time.time()
                drop_or_pop = input("Do you want to drop or pop a piece? [d/p]").strip().lower()
                t_end = time.time()
                if not is_practice_mode and t_end - t_start > READ_TIMEOUT_SECONDS:
                    print(f"Timeout; Please enter your move within {READ_TIMEOUT_SECONDS} seconds.")
                    sys.exit(0)

                if not drop_or_pop in ["d", "p"]:
                    print("Invalid choice.")
                    continue
                drop_or_pop = DROP if drop_or_pop == "d" else POP

                try:
                    column = int(input("Which column? [1-7]"))
                except ValueError:
                    print("Invalid column.")
                    continue
                finally:
                    t_end = time.time()
                    if not is_practice_mode and t_end - t_start > READ_TIMEOUT_SECONDS:
                        print(f"Timeout; Please enter your move within {READ_TIMEOUT_SECONDS} seconds.")
                        sys.exit(0)


                if not game.is_valid_move(column - 1, drop_or_pop, game.current_player):
                    print("Invalid move.")
                    continue

                game.make_move(column - 1, drop_or_pop, game.current_player)

                game.switch_player()
            else:
                # Bot player
                best_action = searcher.search(
                    initialState=State(game, game.current_player),
                )
                
                print(f'Bot chose to {"drop" if best_action[0] == DROP else "pop"} a piece in column {best_action[1] + 1}')
                
                game.make_move(
                    column=best_action[1],
                    mode=best_action[0],
                    player=game.current_player
                )

                game.switch_player()

            print()

        game.print_board()

        if game.winner is not None:
            if game.winner == RED:
                n_human_wins += 1
                print("Congratulations! You won!")
                if n_human_wins == 2:
                    print(f"You won the match. Here is the flag: {flag}")
                    break
            else:
                print("You lost! Better luck next time.")
                # we only play three games; if the human already lost twice, no need to continue
                if n_games_played == 1 and n_human_wins == 0:
                    print("You lost the match.")
                    break


if __name__ == "__main__":
    main()
