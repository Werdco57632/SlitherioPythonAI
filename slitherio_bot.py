import pygame
from game_logic import Slitherio



game = Slitherio()
pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

if hasattr(pygame.display, 'set_subplots'):
    screen = pygame.display.set_subplots()[0]
else:
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

pygame.display.set_caption("Slitherio Bot")

clock = pygame.time.Clock()
FPS = 60

BACKGROUND_COLOR = (30, 30, 40)
PLAYER_COLOR = (0, 255, 150)

running = True
while running:
    # collect Pygame inputs into dict

    keys = pygame.key.get_pressed()
    input_data = {
        'left': keys[pygame.K_LEFT],
        'right': keys[pygame.K_RIGHT],
        'up': keys[pygame.K_UP],
    }



    # calculate next frame using logic file
    state = game.update(input_data)



    # render frame
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (0, 128, 255), (state['x'], state['y'], 50, 50))
    pygame.display.flip()