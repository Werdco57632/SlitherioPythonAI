import pygame
from game_logic import Slitherio
import AI_algorithms
import argparse
import numpy as np
import numpy.typing as npt





class Camera:
    def __init__(self, screen_width, screen_height, map_width, map_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.map_width = map_width
        self.map_height = map_height
        
        # The camera's actual position in the world space
        self.x = 0
        self.y = 0

    def apply(self, world_coord) -> tuple[int, int]:
        """
        Translates a (world_x, world_y) coordinate to a (screen_x, screen_y) coordinate.
        Perfect for direct use in pygame.draw functions.
        """
        world_x, world_y = world_coord
        
        # To get screen position, subtract the camera's position from world position
        screen_x = (world_x - self.x) + (self.screen_width / 2)
        screen_y = (world_y - self.y) + (self.screen_height / 2)
        
        return (int(screen_x), int(screen_y))

    def update(self, target_coord, speed=1) -> None: 
        """
        Moves the camera toward a target (x, y) tuple.
        'speed' controls smoothness (1.0 = instant snap, 0.1 = smooth lag/float).
        """
        target_x, target_y = target_coord

        # Calculate where the camera *wants* to be to center the target on screen
        ideal_x = target_x
        ideal_y = target_y

        # Clamp boundaries based on half-widths/heights so the screen edges don't leak past the map
        max_x_bound = (self.map_width / 2) - (self.screen_width / 2)
        max_y_bound = (self.map_height / 2) - (self.screen_height / 2)

        # Clamp the ideal position so the camera doesn't show outside the map boundaries
        ideal_x = max(-max_x_bound, min(ideal_x, max_x_bound))
        ideal_y = max(-max_y_bound, min(ideal_y, max_y_bound))

        # Linear Interpolation (Lerp) for smooth camera tracking
        # camera_position += (destination - camera_position) * speed
        self.x += (ideal_x - self.x) * speed
        self.y += (ideal_y - self.y) * speed



class player:

    def __init__(self, player_index, game, gamestate):
        self.player_index = player_index
        self.game = game
        self.gamestate = gamestate

        self.algorithm = Algorithm_using(game, player_index) # create AI instance
        self.camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT, MAP_WIDTH, MAP_HEIGHT)
        self.alive = True

        self.update_view()

        

    def update_view(self) -> None:
        snake = self.gamestate.snake_list[self.player_index]
        self.camera.update((snake.x, snake.y))
        self.view = generate_simple_frame(SIMPLE_VIEW_WIDTH, SIMPLE_VIEW_HEIGHT, self.gamestate.food_list, self.gamestate.snake_list, self.camera)

    def get_inputs(self, gamestate) -> dict[str, bool]:
        default_return = {'left': False, 'right': False, 'up': False}
        
        if not self.alive:
            return default_return
        
        self.gamestate = gamestate
        if not self.gamestate.snake_list[self.player_index]:
            self.alive = False
            return default_return
        
        
        self.update_view()
        return self.algorithm.get_inputs(self.gamestate, self.view)





def generate_simple_frame(width, height, food_list, snake_list, camera) -> npt.NDArray[np.float64]:
    simple_frame = np.zeros((width, height), dtype=float)

    pixel_width = camera.screen_width / width
    pixel_height = camera.screen_height / height

    for food in food_list:
        pos = camera.apply((food.x, food.y))
        x = int(pos[0] // pixel_width)
        y = int(pos[1] // pixel_height)
        
        if not (0 <= x < width and 0 <= y < height): # make sure it's on screen
            continue

        size = food.size

        if (simple_frame[x][y] < size):
            simple_frame[x][y] = size/(size+10)



    for snake in snake_list:
        if not snake:
            continue
        
        for i, segment in enumerate(snake.segment_list):
            pos = camera.apply((segment))
            x = int(pos[0] // pixel_width)
            y = int(pos[1] // pixel_height)
            
            if not (0 <= x < width and 0 <= y < height): # make sure it's on screen
                continue

            size = snake.segment_width
            if (i == 0): # make snake head 2x as heavy
                size *= 2



            if (simple_frame[x][y] < size):
                simple_frame[x][y] = size/(size+10)



    return simple_frame



def render_frame(food_list, snake_list, camera) -> None:
    screen.fill(BACKGROUND_COLOR)

    #color_food = (255, 0, 0)
    color_eyes = (255, 255, 255)
    #color_bot = (0, 255, 0)
    #color_player = (0, 255, 255)


    # Draw food
    for food in food_list:
        food_pos = camera.apply((food.x, food.y))
        pygame.draw.circle(screen, food.color, food_pos, food.size)

    # Draw snakes
    for i, snake in enumerate(snake_list):
        if not snake:
            continue

        snake_color = snake.color
        if (i == 0):
            #snake_color = color_player
            print(snake.length)

        for segment in snake.segment_list:
            segment_pos = camera.apply(segment)
            pygame.draw.circle(screen, snake_color, segment_pos, snake.segment_width)

        snake_pos = camera.apply((snake.x, snake.y))
        pygame.draw.circle(screen, color_eyes, snake_pos, snake.segment_width/2)  # Draw head of snake



    pygame.display.flip()

def render_simple_frame(width, height, simple_frame, camera) -> None: # show what the AI sees
    if simple_frame is None:
        return


    # validate frame size
    if (simple_frame.shape != (width, height)):
        raise ValueError(f"can't render simple frame with shape {simple_frame.shape}, should be {(width, height)}")
    
    pixel_width = camera.screen_width / width
    pixel_height = camera.screen_height / height
    


    screen.fill(BACKGROUND_COLOR) # fill extra space just in case

    for i in range(height):
        for j in range(width):
            a = simple_frame[j][i]
            pygame.draw.rect(screen, (a*255, a*255, a*255), (j*pixel_width, i*pixel_height, pixel_width, pixel_height))



    pygame.display.flip()

    return

def gather_inputs(state, player_input_data, AI_players, include_player) -> list[dict[str, bool]]:
    
    input_data = [] # list of dictionaries containing input data for each player

    for i, player in enumerate(AI_players):
        
        if not player: # missing entry in player list
            input_data.append({'left': True, 'right': False, 'up': False}) # NoAI snake
            continue
        


        ai_input_data = AI_players[i].get_inputs(state) # always get the inputs even if unused as it updates the camera and other things

        if include_player and i == 0: # user player is controlling this player
            input_data.append(player_input_data)
            continue

        input_data.append(ai_input_data)



    return input_data







pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

SIMPLE_VIEW_WIDTH = 32
SIMPLE_VIEW_HEIGHT = 24

MAP_WIDTH = 2000
MAP_HEIGHT = 2000

PLAYER_COUNT = 10

if hasattr(pygame.display, 'set_subplots'):
    screen = pygame.display.set_subplots()[0]
else:
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

pygame.display.set_caption("Slitherio Bot")

clock = pygame.time.Clock()
FPS = 60

BACKGROUND_COLOR = (30, 30, 40)
PLAYER_COLOR = (0, 255, 150)






# parse command line arguments
parser = argparse.ArgumentParser(description="Running parameters")
parser.add_argument('-p', '--playerless', action='store_true', help="Exclude player and just have AIs")
parser.add_argument('-g', '--headless', action='store_true', help="Run headlessly; skip rendering (excludes player as well)")
args = parser.parse_args()

Include_player = not args.playerless # whether to let the player play using the active snake
Render_frames = not args.headless # whether to render frames in pygame window
simplify_graphics = False
Algorithm_using = AI_algorithms.Circle_bot # the AI algorythem to use




if (not Render_frames): Include_player = False # don't let the player play if not rendering

Game = Slitherio(width=MAP_WIDTH, height=MAP_HEIGHT, player_count=PLAYER_COUNT)

state = Game.get_state()  # get initial game state





players = [None] * PLAYER_COUNT

for i in range(PLAYER_COUNT):
    players[i] = player(i, Game, state)






running = True
while running:
    
    # collect Pygame inputs into dict

    

    keys = pygame.key.get_pressed()
    player_input_data = {
        'left': keys[pygame.K_LEFT],
        'right': keys[pygame.K_RIGHT],
        'up': keys[pygame.K_UP],
    }

    input_data = gather_inputs(state, player_input_data, players, Include_player)

    # calculate next frame using logic file
    state = Game.update(input_data)

    food_list = state.get_food()
    snake_list = state.get_snakes()




    #region exit conditions
    remaining_players = 0
    winner = None
    winning_length = 0

    for i, snake in enumerate(snake_list):
        if snake:
            remaining_players += 1
            if remaining_players == 1:
                winner = i
                winning_length = snake_list[i].length


    
    if (remaining_players == 1):
        print(f"Winner: player {winner+1} with length {winning_length}. Exiting.")
        running = False
        break

    if (remaining_players == 0):
        print("All snakes are dead. Exiting.")
        running = False
        break

    if (keys[pygame.K_ESCAPE]):
        print("Player pressed ESC. Exiting.")
        running = False
        break

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("Player closed window. Exiting.")
            running = False
            break
    
    #endregion



    if Render_frames: # draw the frame on the screen
        if (simplify_graphics):
            render_simple_frame(SIMPLE_VIEW_WIDTH, SIMPLE_VIEW_HEIGHT, players[0].view, players[0].camera) # show AI view

        else:
            render_frame(food_list, snake_list, players[0].camera) # render everything
        
        #simplify_graphics = not simplify_graphics
        clock.tick(60) # wait for frame to end if drawing



    

