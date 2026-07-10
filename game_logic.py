import random
import math






def generate_spaced_point(existing_objects: list, bounds: tuple[tuple[float, float], tuple[float, float]], k=10) -> tuple[float, float]:
    """
    Generates a new point within bounds that is generally far from existing_objects.
    
    bounds: Tuple of ((min_x, max_x), (min_y, max_y))
    existing_objects: List of objects with x and y attributes [(obj1), (obj2), ...]
    k: Number of candidate points to test (higher = better spacing, lower = faster)
    """

    (min_x, max_x), (min_y, max_y) = bounds
    
    # If it's the very first point, just pick a random spot
    if not existing_objects:
        return (random.uniform(min_x, max_x), random.uniform(min_y, max_y))
    
    best_candidate = None
    best_distance = -1
    
    # Test k random candidates
    for _ in range(k):
        candidate = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        
        # Find the distance to the closest existing point
        min_dist_to_existing = float('inf')
        for obj in existing_objects:
            # squared distance works in place of euclidian distance and is faster
            dist = (candidate[0] - obj.x)**2 + (candidate[1] - obj.y)**2
            if dist < min_dist_to_existing:
                min_dist_to_existing = dist
                
        # Check if this candidate is furthest from its closest neighbor
        if min_dist_to_existing > best_distance:
            best_distance = min_dist_to_existing
            best_candidate = candidate # New best
            
    return best_candidate





class Food:
    def __init__(self, game, state, x, y, size, color=(255,0,0)):
        self.game = game
        self.state = state
        self.x = x
        self.y = y
        self.size = size
        self.color = color

    def update_vacuum(self) -> None:
        
        for snake in self.state.snake_list:
            # Check if the snake's head is close enough to eat this food
            head_x, head_y = snake.segment_list[0]
            d_x = head_x - self.x
            d_y = head_y - self.y
            distance = math.hypot(d_x, d_y)

            

            if (snake.segment_width / 2) < distance < (snake.segment_width / 2 * self.game.food_vacuum_multiplier):
                # Snake is close enough to vacuum the food towards it
                self.x += d_x * self.game.food_vacuum_speed_multiplier * snake.speed
                self.y += d_y * self.game.food_vacuum_speed_multiplier * snake.speed

            if distance < (snake.segment_width / 2):
                # Snake eats the food
                snake.update_length(snake.length + self.size)
                self.state.food_list.remove(self)
                return


class Snake:
    length = 0 # score
    segment_width = 0
    segments = 0
    speed = 0
    heading = 0
    turning_speed = 0
    sprint_time = 0

    

    def __init__(self, game, state, x, y, color=(255,0,0)):
        self.game = game
        self.state = state
        self.x = x
        self.y = y

        self.color = color

        self.segment_list = [(0,0)] * self.game.min_segments
        self.update_length(self.game.min_length)

    


    def update_length(self, new_length) -> None | tuple[float, float]:
        self.length = new_length
        self.segments = len(self.segment_list)
        num_segments_old = self.segments

        self.segments = (new_length - self.game.min_segments) * self.game.growthrate_segments + self.game.min_segments
        self.segment_width = (new_length - self.game.min_segment_width) * self.game.growthrate_segment_width + self.game.min_segment_width
        self.turning_speed = (new_length - self.game.min_turning_speed) * self.game.growthrate_turning_speed + self.game.min_turning_speed
        self.speed = (new_length - self.game.min_speed) * self.game.growthrate_speed + self.game.min_speed

        tail_segment = self.segment_list[-1]

        if (num_segments_old < self.segments): # grew
            self.segment_list.append(self.segment_list[-1])

        elif (num_segments_old > self.segments): # shrunk
            self.segment_list.pop()

        return tail_segment
            
        


    def update_position(self, input_turn, input_sprint) -> None:
        self.update_length(self.length)
        
        clamped_turn = max(-self.turning_speed, min(self.turning_speed, input_turn))
        self.heading += clamped_turn
        self.heading = self.heading % 360

        if (input_sprint and self.length > self.game.min_length and self.sprint_time <= 0):
            self.sprint_time = self.game.sprint_per_length
            tail_segment = self.update_length(self.length - 1) # lose length when sprinting
            if (tail_segment and random.random() < self.game.sprint_drop_chance):
                self.state.create_food(tail_segment[0], tail_segment[1], 1, color=self.color) # drop the segment as food (with chance)

        if (self.sprint_time > 0):
            self.sprint_time -= 1



        # determine new head position
        adjusted_speed = self.speed * self.game.sprint_multiplier if self.sprint_time > 0 else self.speed
        new_x = self.x + adjusted_speed * math.cos(math.radians(self.heading))
        new_y = self.y + adjusted_speed * math.sin(math.radians(self.heading))



        # Handle collisions and death conditions
        if (new_x < -self.game.width/2 or new_x > self.game.width/2 or new_y < -self.game.height/2 or new_y > self.game.height/2):
            self.kill_self(drop_food=False) # don't drop food if you leave the map
            return

        collision_snake = self.state.snake_collision_check(new_x, new_y, self.segment_width / 2, exclude_snake=self)
        if (collision_snake != None):
            self.kill_self()
            return


        # passed collision checks
        self.x = new_x
        self.y = new_y

        # move each piece in the chain
        for i in range(len(self.segment_list)):
            if (i == 0): # move head
                self.segment_list[i] = (new_x, new_y)
            else:
                prev_segment = self.segment_list[i-1]
                curr_segment = self.segment_list[i]

                # Calculate the distance between the current segment and the previous one
                dx = prev_segment[0] - curr_segment[0]
                dy = prev_segment[1] - curr_segment[1]
                distance = math.hypot(dx, dy)

                if distance <= self.segment_width * self.game.segment_chain_multiplier:
                    continue  # Segment is good
                
                # Clamp the segment into the chain distance of the previous segment

                scale = self.segment_width * self.game.segment_chain_multiplier / distance

                new_x = prev_segment[0] - (dx * scale)
                new_y = prev_segment[1] - (dy * scale)

                self.segment_list[i] = (new_x, new_y)






    def kill_self(self, drop_food=True) -> None:
        
        if (drop_food):
            value_to_drop = self.length * self.game.death_drop_chance + self.game.food_bonus
            food_to_drop = math.floor(value_to_drop / self.game.food_size_death)
            segment_diameter = self.segment_width * self.game.segment_chain_multiplier

            for _ in range(food_to_drop):
                random_segment = random.choice(self.segment_list)
                self.state.create_food_random(random_segment[0], random_segment[1], segment_diameter, segment_diameter, self.game.food_size_death, color=self.color)
        
        self.state.snake_list.remove(self)



class Gamestate:


    def __init__(self, game, width: float, height: float, food_count: int, snake_count: int):
        self.game = game

        self.food_list = [] # list of all food objects
        self.snake_list = [] # list of all snake objects

        for _ in range(food_count):
            self.create_food_random(0, 0, width, height)
        for _ in range(snake_count):
            self.create_snake_random(0, 0, width, height)


    

    def create_food(self, x: float, y: float, size: float, color=None) -> None:
        if color == None:
            color = random.choice(self.game.color_list)
        self.food_list.append(Food(self.game, self, x, y, size, color))

    def create_food_random(self, x: float, y: float, width: float, height: float, size=None, color=None) -> None:
        bounds = ((x-width/2, x+width/2),(y-height/2, y+height/2))
        new_food_pos = generate_spaced_point(self.food_list, bounds)

        if size is None: # if no size is specified, choose a random size from the game's food_sizes list
            size = random.choice(self.game.food_sizes)
        if color == None:
            color = random.choice(self.game.color_list)

        new_food = Food(self.game, self, new_food_pos[0], new_food_pos[1], size, color)
        self.food_list.append(new_food)



    def create_snake(self, x: float, y: float, size: float, color=None) -> None:
        if color == None:
            color = random.choice(self.game.color_list)
        self.snake_list.append(Snake(self.game, self, x, y, size, color))

    def create_snake_random(self, x: float, y: float, width: float, height: float, color=None) -> None:
        bounds = ((x-width/2, x+width/2),(y-height/2, y+height/2))
        new_snake_pos = generate_spaced_point(self.snake_list, bounds)

        if color == None:
            color = random.choice(self.game.color_list)

        new_snake = Snake(self.game, self, new_snake_pos[0], new_snake_pos[1], color)
        self.snake_list.append(new_snake)



    def snake_collision_check(self, x: float, y: float, radius: float, exclude_snake=None) -> Snake | None:
        for snake in self.snake_list:
            if snake == exclude_snake:
                continue
            for segment in snake.segment_list:
                if (math.hypot(x - segment[0], y - segment[1]) < snake.segment_width / 2 + radius):
                    return snake
        return None
    


    def ensure_min_food(self, min_food: int) -> None:
        while len(self.food_list) < min_food:
            self.create_food_random(0, 0, self.game.width, self.game.height)

    



    def update(self, inputs: list[tuple[bool, bool, bool]]) -> None:
        # Update all snakes based on input
        for i, snake in enumerate(self.snake_list):
            snake.update_position(inputs[i][1]*90-inputs[i][0]*90, inputs[i][2]) # update_position(turn_input, sprint_input)

        # Update all food (for vacuuming)
        for food in self.food_list:
            food.update_vacuum()

        # Ensure there's always a minimum amount of food on the map
        self.ensure_min_food(self.game.food_max)


    # return functions

    def get_food(self) -> list[Food]:
        return self.food_list

    def get_snakes(self) -> list[Snake]:
        return self.snake_list





class Slitherio:
    
    # config
    sprint_drop_chance = 0.9 # (0-1) chance food is left behind when a snake sprints
    death_drop_chance = 0.6 # (0-1) chance food is left behind when a snake dies
    food_sizes = [1,2,4] # distribution of random food on the ground
    food_size_death = 9 # size of food dropped when a snake dies
    food_bonus = 50 # amount of food to drop dispite snake's length

    min_length = 10 # minimum length of a snake
    
    min_segment_width = 10 # minimum width of a snake segment
    growthrate_segment_width = 0.005 # how much the segment width grows per length

    segment_chain_multiplier = 0.8 # (0-1) how much of a segment width each segment is spaced away from the next segment in the chain
    food_vacuum_multiplier = 5 # how much of a segment width the snake can vacuum food from
    food_vacuum_speed_multiplier = 0.1 # how many pixels per frame the food moves towards the snake when vacuumed relitive to the snake's speed

    min_segments = 10 # minimum amount of snake segments
    growthrate_segments = 0.05 # how many segments are added per length

    min_turning_speed = 4 # minimum turning speed (degrees per frame)
    growthrate_turning_speed = -0.0001 # how much the turning speed grows per length

    min_speed = 2 # minimumpixels per frame speed of a snake
    growthrate_speed = 0.001 # how much the speed grows per length
    
    sprint_multiplier = 1.5 # multiplier of a snake's speed while sprinting
    sprint_per_length = 30 # how many frames you can sprint per length used

    color_list = [(200,10,10),(10,200,10),(10,10,200),(200,200,10),(200,10,200),(10,200,200)]




    def __init__(self, width=1000, height=1000, food_max=1000, player_count=10):
        self.width = width
        self.height = height
        self.food_max = food_max
        self.player_count = player_count

        self.current_state = self.start_state()



    

    def start_state(self) -> Gamestate: # returns a defaultly instantiated gamestate
        return Gamestate(self, self.width, self.height, self.food_max, self.player_count)

    
    
    def get_state(self) -> Gamestate:
        return self.current_state

    def update(self, keys_pressed: list[dict[str, bool]]) -> Gamestate:
        """
        Takes raw input (from the main file) and 
        returns the updated state.
        """
        

        self.current_state.update(inputs=[(keys['left'], keys['right'], keys['up']) for keys in keys_pressed])

        return self.current_state
    

