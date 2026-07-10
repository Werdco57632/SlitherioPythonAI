import random

class Slitherio:
    
    def __init__(self, width=100, height=100, food_max=1000):
        self.width = width
        self.height = height
        self.food_max = food_max



    



    class Gamestate:

        food_list = [] # array of food coordinates

        class Food:
            def __init__(self, x, y, size):
                self.x = x
                self.y = y
                self.size = size
            
        def create_food(self, x, y, size):
            self.food_list.append(self.Food(x, y, size))

        def create_food_random(self, width, height):
            food_sizes = [1,2,4]
            new_food = self.Food(random.uniform(-width/2, width/2), random.uniform(-height/2, height/2), random.choice(food_sizes))
            self.food_list.append(new_food)


        
        class Snake:
            length = 0 # score
            segment_width = 0
            segments = 0
            speed = 0
            heading = 0
            turning_speed = 4

            segment_spacing = 0.2
            
            segments = [(0,0)] * 10

            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.update_length(10)
            
            def update_length(self, new_length): # returns Tupple of food to be created or None
                self.length = new_length
                self.segment_width = (new_length-20)*0.1+20
                segments_old = self.segments
                self.segments = (new_length-10)*0.25+10

                if (segments_old < self.segments): # grew
                    self.segments.append(self.segments[-1])

                elif (segments_old > self.segments): # shrunk
                    
                    return self.segments.pop()
                
                return None

            def move(turn, sprint):
                heading += turn
                heading = heading % 360
                


        

        def __init__(self, width, height, food_count, enemy_count):
            for _ in range(food_count):
                self.create_food_random(width, height)



    def __init__(self):
        self.current_state = self.start_state()

    def start_state(self): # returns a defaultly instantiated gamestate
        return self.Gamestate(100,100,1000,10)

    
    
    def update(self, keys_pressed):
        """
        Takes raw input (from the main file) and 
        returns the updated state.
        """
        if keys_pressed['left']: self.x -= self.speed
        if keys_pressed['right']: self.x += self.speed
        if keys_pressed['up']: self.y -= self.speed
        if keys_pressed['down']: self.y += self.speed
        
        # Return state as a dictionary or object for rendering
        return {"x": self.x, "y": self.y}
    

    