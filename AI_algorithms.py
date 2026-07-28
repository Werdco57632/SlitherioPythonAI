import math
from game_logic import Slitherio
import heapq




class Circle_bot:
    def __init__(self, game, player_id):
        self.game = game
        self.state = None
        self.player_id = player_id

    def get_inputs(self, state, view):
        self.state = state
        self.snake = state.snake_list[self.player_id]

        return {'left': True, 'right': False, 'up': False}
    


class Reflex_bot:
    """
    A greedy reflex agent: no lookahead/search, just a per-frame scoring of
    "where should I steer right now" based on the current game state.

    Each frame it:
      1. Builds an attraction vector toward the nearest food.
      2. Builds repulsion vectors away from nearby danger (other snakes'
         bodies, and the map walls), stronger the closer the danger is.
      3. Sums these into one desired-direction vector.
      4. Turns left/right to steer the snake's heading toward that vector.
      5. Sprints only when nothing dangerous is nearby.
    """

    # --- tunable weights/thresholds ---
    FOOD_WEIGHT = 1.0

    SNAKE_DANGER_RADIUS = 150    # how far away another snake's body starts to matter
    SNAKE_DANGER_WEIGHT = 400    # how strongly it pushes us away

    WALL_MARGIN = 120            # start avoiding walls this close to the edge
    WALL_WEIGHT = 300

    TURN_DEADZONE = 5            # degrees; don't bother turning for tiny corrections
    SPRINT_MIN_DANGER_DIST = 200 # only sprint if nothing dangerous is closer than this

    def __init__(self, game, player_id):
        self.game = game
        self.state = None
        self.player_id = player_id

    def get_inputs(self, state, view):
        self.state = state
        self.snake = state.snake_list[self.player_id]

        head_x, head_y = self.snake.segment_list[0]

        target_dx, target_dy = 0.0, 0.0
        nearest_danger_dist = float('inf')

        # 1. Attraction toward the nearest food
        nearest_food = None
        nearest_food_dist = float('inf')
        for food in state.food_list:
            d = math.hypot(food.x - head_x, food.y - head_y)
            if d < nearest_food_dist:
                nearest_food_dist = d
                nearest_food = food

        if nearest_food is not None and nearest_food_dist > 0:
            dx = nearest_food.x - head_x
            dy = nearest_food.y - head_y
            target_dx += self.FOOD_WEIGHT * dx / nearest_food_dist
            target_dy += self.FOOD_WEIGHT * dy / nearest_food_dist

        # 2. Repulsion from other snakes' segments
        for other in state.snake_list:
            if not other or other is self.snake:
                continue
            for seg_x, seg_y in other.segment_list:
                d = math.hypot(seg_x - head_x, seg_y - head_y)
                if d < self.SNAKE_DANGER_RADIUS:
                    nearest_danger_dist = min(nearest_danger_dist, d)
                    d = max(d, 1.0)  # avoid divide-by-zero when segments overlap the head
                    push = self.SNAKE_DANGER_WEIGHT / (d * d)
                    target_dx -= push * (seg_x - head_x) / d
                    target_dy -= push * (seg_y - head_y) / d

        # 3. Repulsion from walls
        half_w = self.game.width / 2
        half_h = self.game.height / 2

        wall_distances = (
            (half_w - head_x, (-1, 0)),   # right wall pushes left
            (head_x + half_w, (1, 0)),    # left wall pushes right
            (half_h - head_y, (0, -1)),   # bottom wall pushes up
            (head_y + half_h, (0, 1)),    # top wall pushes down
        )
        for dist, direction in wall_distances:
            if dist < self.WALL_MARGIN:
                nearest_danger_dist = min(nearest_danger_dist, dist)
                dist = max(dist, 1.0)
                push = self.WALL_WEIGHT / (dist * dist)
                target_dx += push * direction[0]
                target_dy += push * direction[1]

        # 4. Steer toward the desired direction
        if target_dx == 0 and target_dy == 0:
            # nothing pulling us anywhere in particular; just keep going straight
            return {'left': False, 'right': False, 'up': True}

        desired_heading = math.degrees(math.atan2(target_dy, target_dx)) % 360
        heading_diff = (desired_heading - self.snake.heading + 180) % 360 - 180

        turn_left = False
        turn_right = False
        if heading_diff > self.TURN_DEADZONE:
            turn_right = True
        elif heading_diff < -self.TURN_DEADZONE:
            turn_left = True

        # 5. Sprint only when it's safe to do so
        sprint = nearest_danger_dist > self.SPRINT_MIN_DANGER_DIST

        return {'left': turn_left, 'right': turn_right, 'up': sprint}


class AStar_bot:
    """
    Plans a route to the nearest food with A* search over a discretized grid,
    then steers along that route frame by frame, replanning periodically
    since other snakes move and make old plans stale.
    """

    CELL_SIZE = 40              # grid resolution in world pixels
    REPLAN_INTERVAL = 20        # frames between replans
    WAYPOINT_REACH_DIST = 30    # how close (pixels) counts as "reached" a waypoint
    OBSTACLE_INFLATION = 1.5    # safety margin around snake bodies, in units of snake radius
    TURN_DEADZONE = 5           # degrees
    WALL_MARGIN = 150          # start penalizing cells within this distance of a wall
    WALL_PENALTY_WEIGHT = 50   # how strongly it discourages wall-hugging routes

    def __init__(self, game, player_id):
        self.game = game
        self.state = None
        self.player_id = player_id
        self.path = []          # list of (x, y) waypoints, world coords
        self.frames_since_plan = 0

    def _wall_penalty(self, cell):
        """Extra pathing cost for cells close to a wall -- discourages
        hugging the boundary without forbidding it outright (food can
        still spawn right against a wall)."""
        x, y = self._to_world(cell)
        half_w, half_h = self.game.width / 2, self.game.height / 2
        dist_to_wall = min(half_w - abs(x), half_h - abs(y))

        if dist_to_wall >= self.WALL_MARGIN:
            return 0.0

        dist_to_wall = max(dist_to_wall, 1.0)
        return self.WALL_PENALTY_WEIGHT / dist_to_wall

    def get_inputs(self, state, view):
        self.state = state
        self.snake = state.snake_list[self.player_id]
        head_x, head_y = self.snake.segment_list[0]

        self.frames_since_plan += 1
        need_replan = not self.path or self.frames_since_plan >= self.REPLAN_INTERVAL

        if need_replan:
            self._plan_path(head_x, head_y)
            self.frames_since_plan = 0

        # drop waypoints we've already reached
        while self.path and math.hypot(self.path[0][0] - head_x, self.path[0][1] - head_y) < self.WAYPOINT_REACH_DIST:
            self.path.pop(0)

        if not self.path:
            # No planned path -- either nothing to plan to, or the fresh
            # plan's only waypoint was already within reach and got popped
            # above. Don't just cruise straight blind; steer directly at
            # the nearest food as a safety net instead.
            food = self._nearest_food(head_x, head_y)
            if food is None:
                return {'left': False, 'right': False, 'up': False}

            desired_heading = math.degrees(math.atan2(food.y - head_y, food.x - head_x)) % 360
            heading_diff = (desired_heading - self.snake.heading + 180) % 360 - 180
            turn_left = heading_diff < -self.TURN_DEADZONE
            turn_right = heading_diff > self.TURN_DEADZONE
            return {'left': turn_left, 'right': turn_right, 'up': False}

        target_x, target_y = self.path[0]
        desired_heading = math.degrees(math.atan2(target_y - head_y, target_x - head_x)) % 360
        heading_diff = (desired_heading - self.snake.heading + 180) % 360 - 180

        turn_left = heading_diff < -self.TURN_DEADZONE
        turn_right = heading_diff > self.TURN_DEADZONE

        return {'left': turn_left, 'right': turn_right, 'up': False}

    def _plan_path(self, head_x, head_y):
        target = self._nearest_food(head_x, head_y)
        if target is None:
            self.path = []
            return

        obstacles = self._build_obstacle_grid()
        start_cell = self._to_cell(head_x, head_y)
        goal_cell = self._to_cell(target.x, target.y)

        cell_path = self._a_star(start_cell, goal_cell, obstacles)
        self.path = [self._to_world(c) for c in cell_path] if cell_path else []

    def _nearest_food(self, head_x, head_y):
        nearest, nearest_dist = None, float('inf')
        for food in self.state.food_list:
            d = math.hypot(food.x - head_x, food.y - head_y)
            if d < nearest_dist:
                nearest_dist, nearest = d, food
        return nearest

    def _to_cell(self, x, y):
        return (int(x // self.CELL_SIZE), int(y // self.CELL_SIZE))

    def _to_world(self, cell):
        cx, cy = cell
        return (cx * self.CELL_SIZE + self.CELL_SIZE / 2, cy * self.CELL_SIZE + self.CELL_SIZE / 2)

    def _build_obstacle_grid(self):
        obstacles = set()
        for other in self.state.snake_list:
            if not other or other is self.snake:
                continue
            radius_cells = math.ceil((other.segment_width / 2 * self.OBSTACLE_INFLATION) / self.CELL_SIZE)
            for seg_x, seg_y in other.segment_list:
                seg_cell = self._to_cell(seg_x, seg_y)
                for dx in range(-radius_cells, radius_cells + 1):
                    for dy in range(-radius_cells, radius_cells + 1):
                        obstacles.add((seg_cell[0] + dx, seg_cell[1] + dy))
        return obstacles

    def _in_bounds(self, cell):
        x, y = self._to_world(cell)
        half_w, half_h = self.game.width / 2, self.game.height / 2
        return -half_w <= x <= half_w and -half_h <= y <= half_h

    def _a_star(self, start, goal, obstacles):
        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_set = [(heuristic(start, goal), 0, start)]
        came_from = {}
        g_score = {start: 0}
        visited = set()
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        while open_set:
            _, g, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in obstacles or neighbor in visited or not self._in_bounds(neighbor):
                    continue

                tentative_g = g + math.hypot(dx, dy) + self._wall_penalty(neighbor)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (tentative_g + heuristic(neighbor, goal), tentative_g, neighbor))

        return None  # no path found
    
class Hunter_bot:
    
    """ Actively hunts opponents by predicting their future position (lead
    pursuit, same idea as Pac-Man's Pinky ambushing ahead of the player)
    and steering to get in their way -- since a snake dies when ITS OWN
    head touches another snake's body, "attacking" just means positioning
    yourself in their predicted path. Falls back to Reflex_bot-style
    food-seeking when nothing worth hunting is nearby.
    """

    HUNT_RADIUS = 400
    LOOKAHEAD_FRAMES = 25
    INTERCEPT_WEIGHT = 1.5
    FOOD_WEIGHT = 0.5

    SNAKE_DANGER_RADIUS = 150
    SNAKE_DANGER_WEIGHT = 400
    WALL_MARGIN = 250
    WALL_WEIGHT = 1200
    TURN_DEADZONE = 5
    SPRINT_MIN_DANGER_DIST = 200

    def __init__(self, game, player_id):
        self.game = game
        self.state = None
        self.player_id = player_id

    def get_inputs(self, state, view):
        self.state = state
        self.snake = state.snake_list[self.player_id]
        head_x, head_y = self.snake.segment_list[0]

        target_dx, target_dy = 0.0, 0.0
        nearest_danger_dist = float('inf')

        target_snake = self._choose_target(head_x, head_y)

        if target_snake is not None:
            intercept_x, intercept_y = self._predict_intercept(target_snake)
            d = math.hypot(intercept_x - head_x, intercept_y - head_y)
            if d > 0:
                target_dx += self.INTERCEPT_WEIGHT * (intercept_x - head_x) / d
                target_dy += self.INTERCEPT_WEIGHT * (intercept_y - head_y) / d
        else:
            food = self._nearest_food(head_x, head_y)
            if food is not None:
                d = math.hypot(food.x - head_x, food.y - head_y)
                if d > 0:
                    target_dx += self.FOOD_WEIGHT * (food.x - head_x) / d
                    target_dy += self.FOOD_WEIGHT * (food.y - head_y) / d

        # repulsion from every other snake's body -- get close, don't touch
        for other in state.snake_list:
            if not other or other is self.snake:
                continue
            for seg_x, seg_y in other.segment_list:
                d = math.hypot(seg_x - head_x, seg_y - head_y)
                if d < self.SNAKE_DANGER_RADIUS:
                    nearest_danger_dist = min(nearest_danger_dist, d)
                    d = max(d, 1.0)
                    push = self.SNAKE_DANGER_WEIGHT / (d * d)
                    target_dx -= push * (seg_x - head_x) / d
                    target_dy -= push * (seg_y - head_y) / d

        # repulsion from walls
        half_w, half_h = self.game.width / 2, self.game.height / 2
        wall_distances = (
            (half_w - head_x, (-1, 0)),
            (head_x + half_w, (1, 0)),
            (half_h - head_y, (0, -1)),
            (head_y + half_h, (0, 1)),
        )
        for dist, direction in wall_distances:
            if dist < self.WALL_MARGIN:
                nearest_danger_dist = min(nearest_danger_dist, dist)
                dist = max(dist, 1.0)
                push = self.WALL_WEIGHT / (dist * dist)
                target_dx += push * direction[0]
                target_dy += push * direction[1]

        if target_dx == 0 and target_dy == 0:
            return {'left': False, 'right': False, 'up': True}

        desired_heading = math.degrees(math.atan2(target_dy, target_dx)) % 360
        heading_diff = (desired_heading - self.snake.heading + 180) % 360 - 180

        turn_left = heading_diff < -self.TURN_DEADZONE
        turn_right = heading_diff > self.TURN_DEADZONE
        sprint = nearest_danger_dist > self.SPRINT_MIN_DANGER_DIST

        return {'left': turn_left, 'right': turn_right, 'up': sprint}

    def _choose_target(self, head_x, head_y):
        nearest, nearest_dist = None, float('inf')
        for other in self.state.snake_list:
            if not other or other is self.snake:
                continue
            d = math.hypot(other.x - head_x, other.y - head_y)
            if d < self.HUNT_RADIUS and d < nearest_dist:
                nearest_dist, nearest = d, other
        return nearest

    def _predict_intercept(self, target_snake):
        rad = math.radians(target_snake.heading)
        predicted_x = target_snake.x + math.cos(rad) * target_snake.speed * self.LOOKAHEAD_FRAMES
        predicted_y = target_snake.y + math.sin(rad) * target_snake.speed * self.LOOKAHEAD_FRAMES

        half_w = self.game.width / 2
        half_h = self.game.height / 2
        predicted_x = max(-half_w + 40, min(half_w - 40, predicted_x))
        predicted_y = max(-half_h + 40, min(half_h - 40, predicted_y))
        return predicted_x, predicted_y

    def _nearest_food(self, head_x, head_y):
        nearest, nearest_dist = None, float('inf')
        for food in self.state.food_list:
            d = math.hypot(food.x - head_x, food.y - head_y)
            if d < nearest_dist:
                nearest_dist, nearest = d, food
        return nearest
# TODO make a better algorithm