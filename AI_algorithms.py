import math
from game_logic import Slitherio





class Circle_bot:
    def __init__(self, game, player_id):
        self.game = game
        self.state = None
        self.player_id = player_id

    def get_inputs(self, state, view):
        self.state = state
        self.snake = state.snake_list[self.player_id]

        return {'left': True, 'right': False, 'up': False}
    



# TODO make a better algorithm