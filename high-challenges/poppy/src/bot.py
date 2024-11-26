import typing

from game import Game, EMPTY, RED, YELLOW, ROWS, COLUMNS, DROP, POP

class State:

    def __init__(self, game: 'Game', me: int):
        self.game = game
        self.me = me
    
    def getCurrentPlayer(self):
        if self.game.current_player == self.me:
            return 1
        else:
            return -1
    
    def getPossibleActions(self):
        state = self.game.state
        current_player = self.game.current_player
        actions = []
        assert current_player in (RED, YELLOW)
        actions = []

        # Where can we drop a piece?
        for column in range(COLUMNS):
            if state[0][column] == EMPTY:
                actions.append((DROP, column))

        # Where can we pop a piece?
        for column in range(COLUMNS):
            if state[ROWS - 1][column] == current_player:
                actions.append((POP, column))
    
        return actions

    def takeAction(self, action: typing.Tuple[int, int]):
        action_type, column = action
        game_cpy = self.game.copy()

        game_cpy.make_move(
            column=column,
            mode=action_type,
            player=self.game.current_player
        )
        game_cpy.switch_player()

        return State(game_cpy, self.me)

    def isTerminal(self):
        return self.game.winner is not None

    def getReward(self):
        if self.game.winner is None:
            return 0
        elif self.game.winner != self.me:
            return -1
        elif self.game.winner == self.me:
            return 1
