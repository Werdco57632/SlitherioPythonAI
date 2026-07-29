"""
AI_algorithms.py -- Agent decision-making logic.
CAI 4002 group project.

This file contains two independently developed families of agents, kept together
so the evaluation framework can compare them directly.

  Potential-field / A* family
      Reflex_bot    -- potential fields: attraction to food, repulsion from danger
      AStar_bot     -- A* over a discretised grid, replanned periodically
      Hunter_bot    -- lead pursuit, intercepts a target's predicted position

  Forward-simulation family
      Wall_bot      -- survival-only baseline
      Greedy_bot    -- pure food-seeking reflex, no danger model
      Reactive_bot  -- prioritised rule ladder (walls > snakes > food)
      Lookahead_bot -- depth-limited forward simulation of the game's own physics
      Aggressive_bot-- same search core, retargeted onto smaller opponents
      Neural_bot    -- adapter for the evolutionary CNN agents

  Circle_bot        -- original placeholder, kept as the performance floor

All agents share the interface established by Circle_bot:

    class MyBot:
        def __init__(self, game, player_id): ...
        def get_inputs(self, state, view) -> dict[str, bool]   # keys: left, right, up

Select one in slitherio_render.py:

    Algorithm_using = AI_algorithms.Lookahead_bot

--------------------------------------------------------------------------
ENVIRONMENT NOTES THAT DRIVE THESE DESIGNS
--------------------------------------------------------------------------
* Continuous space, not a grid. A snake has a float position (x, y) and a
  `heading` in degrees; each frame it advances `speed` pixels and may turn by at
  most `turning_speed` degrees. Actions are therefore *steering* commands, not
  destination choices, so grid search needs an explicit discretisation step
  (see AStar_bot) or must be replaced by simulation (see Lookahead_bot).
* The action space is three independent booleans. game_logic converts them to
  a turn of (right - left) * 90 degrees, clamped to +/-turning_speed, so
  pressing both left and right is equivalent to going straight.
* Self-collision is impossible: Snake.update_position calls
  snake_collision_check(..., exclude_snake=self). A snake can only die by
  leaving the map or touching ANOTHER snake.
* Sprinting ('up') costs one length per activation, with a cooldown of
  game.sprint_per_length frames, and 90% of that length drops as food for
  opponents. At 60 FPS, holding sprint drains roughly 2 length/second.
* Food is eaten when the head is within segment_width / 2 of it -- about 5px
  at minimum length. Any planner that navigates to a coarser target (a grid
  cell centre, say) must add a final approach to the food's true position.
--------------------------------------------------------------------------

MEASURED OBSERVATIONS (see headless_eval.py to reproduce)
--------------------------------------------------------------------------
Three findings worth acting on, recorded here rather than silently patched:

1. AStar_bot under-eats. Its final waypoint is the food's grid CELL CENTRE,
   which sits up to 28px from the food, and WAYPOINT_REACH_DIST pops that
   waypoint at 30px -- but eating requires 5px. It can therefore complete its
   path without eating. Appending (target.x, target.y) as a final waypoint
   raised average peak length from 19.9 to 25.1 over 800 frames.

2. Reflex_bot and Hunter_bot sprint by default. `nearest_danger_dist > 200` is
   true most of the time in open space, and each activation costs length, so
   they bleed roughly 2 length/second while safe. Inverting this -- sprint only
   when there is a reason to -- should raise both agents' growth.

3. Wall deaths dominate for long snakes. growthrate_turning_speed is NEGATIVE
   and segment_width grows with length, so a long snake is wider AND turns
   slower; its minimum turning radius is speed / radians(turning_speed).
   Escaping a wall it is driving at costs about twice that radius, so wall cost
   has to scale with size. Applying this to Lookahead_bot took its survival
   rate from 33% to 56%. Agents whose fallback is "go straight" when they have
   no plan (AStar_bot) are most exposed to this.
--------------------------------------------------------------------------
"""

import math
import heapq

from game_logic import Slitherio

# Discretised action space for the turn component.
TURN_LEFT = -1
TURN_STRAIGHT = 0
TURN_RIGHT = 1
TURNS = (TURN_LEFT, TURN_STRAIGHT, TURN_RIGHT)

NO_INPUT = {'left': False, 'right': False, 'up': False}

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class Circle_bot:
    """Original placeholder: turns left forever. Establishes the performance floor."""

    name = "circle"

    def __init__(self, game, player_id):
        self.game = game
        self.state = None
        self.player_id = player_id

    def get_inputs(self, state, view=None):
        self.state = state
        self.snake = state.snake_list[self.player_id]
        return {'left': True, 'right': False, 'up': False}


# ===========================================================================
# Potential-field and grid-search family
# ---------------------------------------------------------------------------
# Preserved as written. See MEASURED OBSERVATIONS 1 and 2 above.
# ===========================================================================

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


# ===========================================================================
# Forward-simulation family
# ===========================================================================

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def angle_to(origin, target):
    """Absolute heading in degrees from origin to target."""
    return math.degrees(math.atan2(target[1] - origin[1], target[0] - origin[0]))


def angle_error(desired, current):
    """Signed smallest rotation from current heading to desired, in [-180, 180]."""
    return (desired - current + 180.0) % 360.0 - 180.0


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def sq_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def turn_toward(error, deadzone=1.0):
    """Bang-bang controller: which discrete turn reduces the heading error."""
    if error > deadzone:
        return TURN_RIGHT
    if error < -deadzone:
        return TURN_LEFT
    return TURN_STRAIGHT


def turn_to_inputs(turn, sprint=False):
    return {
        'left': turn == TURN_LEFT,
        'right': turn == TURN_RIGHT,
        'up': bool(sprint),
    }


class SteeringBot:
    """
    Common machinery for all hand-designed agents: locates the agent's own snake,
    builds a local picture of nearby food, opponent bodies and opponent heads,
    and converts a chosen turn into the environment's input dict.
    """

    name = "steering"

    # perception ranges, in pixels
    food_scan_radius = 400.0
    danger_scan_radius = 260.0

    def __init__(self, game, player_id):
        self.game = game
        self.player_id = player_id
        self.state = None
        self.snake = None
        self.frames = 0

    # -- helpers ------------------------------------------------------------

    @property
    def bounds(self):
        """(min_x, max_x, min_y, max_y) of the playable map."""
        return (-self.game.width / 2.0, self.game.width / 2.0,
                -self.game.height / 2.0, self.game.height / 2.0)

    def my_snake(self, state):
        snakes = state.snake_list
        if self.player_id >= len(snakes):
            return None
        return snakes[self.player_id]

    def nearby_food(self, head, radius=None):
        """Food within `radius` as a list of (x, y, size, dist)."""
        radius = self.food_scan_radius if radius is None else radius
        r2 = radius * radius
        out = []
        for food in self.state.food_list:
            d2 = sq_distance(head, (food.x, food.y))
            if d2 <= r2:
                out.append((food.x, food.y, food.size, math.sqrt(d2)))
        return out

    def nearby_threats(self, head, radius=None):
        """
        Local obstacle picture for every OTHER snake.

        Returns (segments, heads) where
          segments = [(x, y, radius), ...]                     body segments
          heads    = [(x, y, vx, vy, radius, length), ...]     projected heads
        """
        radius = self.danger_scan_radius if radius is None else radius
        r2 = radius * radius
        segments = []
        heads = []

        for i, snake in enumerate(self.state.snake_list):
            if not snake or i == self.player_id:
                continue

            seg_radius = snake.segment_width / 2.0
            if sq_distance(head, (snake.x, snake.y)) <= (radius + 200.0) ** 2:
                vx = snake.speed * math.cos(math.radians(snake.heading))
                vy = snake.speed * math.sin(math.radians(snake.heading))
                heads.append((snake.x, snake.y, vx, vy, seg_radius, snake.length))

            for seg in snake.segment_list:
                if sq_distance(head, seg) <= r2:
                    segments.append((seg[0], seg[1], seg_radius))

        return segments, heads

    def wall_clearance(self, pos):
        min_x, max_x, min_y, max_y = self.bounds
        return min(pos[0] - min_x, max_x - pos[0], pos[1] - min_y, max_y - pos[1])

    # -- interface ----------------------------------------------------------

    def get_inputs(self, state, view=None):
        self.state = state
        self.frames += 1
        snake = self.my_snake(state)
        if not snake:
            return dict(NO_INPUT)
        self.snake = snake
        return self.decide(snake, view)

    def decide(self, snake, view):
        raise NotImplementedError


class Wall_bot(SteeringBot):
    """
    Minimal survival baseline: drive straight, and turn away only when the map
    edge gets close. Isolates how much of performance is just not dying to walls.
    """

    name = "wall-avoid"
    margin = 120.0

    def decide(self, snake, view):
        head = (snake.x, snake.y)
        min_x, max_x, min_y, max_y = self.bounds

        if self.wall_clearance(head) > self.margin:
            return turn_to_inputs(TURN_STRAIGHT)

        # Steer toward the map centre, which is always a safe direction from an edge.
        error = angle_error(angle_to(head, (0.0, 0.0)), snake.heading)
        return turn_to_inputs(turn_toward(error, deadzone=6.0))


class Greedy_bot(SteeringBot):
    """
    Rule-based reflex agent: steer toward the most attractive nearby food, where
    attractiveness is size discounted by distance. Purely reactive with no
    collision reasoning, so it dies to opponents constantly. This is the
    comparison point that motivates the search agent.
    """

    name = "greedy"

    def best_food(self, head, food):
        best = None
        best_value = -1.0
        for fx, fy, size, dist in food:
            value = size / (dist + 30.0)
            if value > best_value:
                best_value = value
                best = (fx, fy)
        return best

    def decide(self, snake, view):
        head = (snake.x, snake.y)
        food = self.nearby_food(head)
        target = self.best_food(head, food)

        if target is None:
            target = (0.0, 0.0)  # wander toward the middle when nothing is in range

        error = angle_error(angle_to(head, target), snake.heading)
        return turn_to_inputs(turn_toward(error, deadzone=2.0))


class Reactive_bot(Greedy_bot):
    """
    Rule-based agent with a priority hierarchy, in the classic reflex-agent style:

        1. wall avoidance   -- highest priority, walls never move
        2. threat avoidance -- veer away from nearby opponent bodies and from the
                               projected path of any snake that can kill us
        3. food seeking     -- otherwise pursue the best nearby food

    Cheap and interpretable. Its weakness is that each rule reacts to the present
    frame only, so it walks into situations it could have seen coming.
    """

    name = "reactive"
    wall_margin = 140.0
    threat_margin = 90.0

    def decide(self, snake, view):
        head = (snake.x, snake.y)

        # Rule 1: walls
        if self.wall_clearance(head) < self.wall_margin:
            error = angle_error(angle_to(head, (0.0, 0.0)), snake.heading)
            return turn_to_inputs(turn_toward(error, deadzone=4.0))

        segments, heads = self.nearby_threats(head)

        # Rule 2: nearest body segment that is uncomfortably close and roughly ahead
        worst = None
        worst_dist = float('inf')
        for sx, sy, srad in segments:
            d = distance(head, (sx, sy)) - srad - snake.segment_width / 2.0
            if d >= self.threat_margin:
                continue
            bearing = angle_error(angle_to(head, (sx, sy)), snake.heading)
            if abs(bearing) > 100.0:
                continue  # behind us, harmless
            if d < worst_dist:
                worst_dist = d
                worst = bearing

        if worst is not None:
            # Turn away from the obstacle: opposite sign of its bearing.
            return turn_to_inputs(TURN_LEFT if worst > 0 else TURN_RIGHT)

        # Rule 3: food
        return super().decide(snake, view)


# ---------------------------------------------------------------------------
# Search-based agent
# ---------------------------------------------------------------------------

class Lookahead_bot(SteeringBot):
    """
    Search-based agent: depth-limited forward simulation over the discretised
    action space, scored by a heuristic evaluation function.

    Formulation
    -----------
    State:    (position, heading) of our snake, plus a local model of the world.
    Actions:  {left, straight, right} -- the environment's turn commands.
    Model:    replicates game_logic physics -- heading changes by at most
              turning_speed per frame, position advances by speed per frame,
              death on leaving the map or touching another snake's segment.
              Opponent heads are projected forward linearly at their current
              velocity; their bodies are treated as static over the horizon.

    Because a full tree of depth H has 3^H leaves, the search uses *piecewise
    constant plans*: the first `commit_frames` are one turn, the remainder is
    another. That gives 9 rollouts of `horizon` frames each -- cheap enough for
    ten agents at 60 FPS, while still expressing turn, straighten, and
    turn-then-reverse manoeuvres.

    Evaluation (higher is better):
        + frames survived along the rollout, weighted heavily
        + value of food swept up along the way
        + closing distance on the chosen food target
        - proximity to opponents that could kill us
        - proximity to walls
    """

    name = "lookahead"

    horizon = 30            # frames simulated per plan
    commit_frames = 10      # frames the first action is held before switching
    survive_weight = 12.0
    food_weight = 22.0
    progress_weight = 0.9
    threat_weight = 55.0
    body_weight = 18.0
    base_clearance = 30.0
    wall_weight = 16.0
    sprint_min_length_margin = 12.0
    sprint_cooldown = 45
    replan_interval = 3     # frames between searches (plan is held in between)

    def __init__(self, game, player_id):
        super().__init__(game, player_id)
        self._last_sprint = -999
        self._cached = None

    # -- target selection ---------------------------------------------------

    def choose_target(self, snake, food):
        """
        Pick a food to pursue: size discounted by distance, with a mild bonus for
        food that is already ahead of us (turning is the scarce resource here).
        """
        head = (snake.x, snake.y)
        best = None
        best_value = -float('inf')
        for fx, fy, size, dist in food:
            bearing = abs(angle_error(angle_to(head, (fx, fy)), snake.heading))
            value = (size * 12.0) / (dist + 40.0) - bearing / 240.0
            if value > best_value:
                best_value = value
                best = (fx, fy)
        return best

    # -- forward model ------------------------------------------------------

    def rollout(self, snake, plan, segments, heads, food, target):
        """
        Simulate `plan` = (first_turn, second_turn) and return its heuristic score.
        """
        x, y = snake.x, snake.y
        heading = snake.heading
        turn_rate = max(0.5, snake.turning_speed)
        speed = max(0.5, snake.speed)
        my_radius = snake.segment_width / 2.0
        min_x, max_x, min_y, max_y = self.bounds

        eaten = set()
        food_value = 0.0
        threat_penalty = 0.0
        body_penalty = 0.0
        wall_penalty = 0.0
        frames_survived = 0

        # A longer snake is wider and turns slower (growthrate_turning_speed is
        # negative), so it needs more clearance to survive the same manoeuvre.
        clearance = self.base_clearance + my_radius * 1.5

        # Minimum turning radius r = speed / turn_rate (in radians). Escaping a
        # wall you are driving straight at costs roughly 2r of room, so the wall
        # cost has to start applying well before the horizon reaches the edge --
        # otherwise a long, wide, slow-turning snake commits to the boundary
        # before it can see the boundary. Wall deaths dominated before this term.
        turn_radius = speed / max(math.radians(turn_rate), 1e-6)
        wall_range = max(120.0, 2.5 * turn_radius + my_radius + clearance)

        for step in range(self.horizon):
            turn = plan[0] if step < self.commit_frames else plan[1]
            heading = (heading + turn * turn_rate) % 360.0
            rad = math.radians(heading)
            x += speed * math.cos(rad)
            y += speed * math.sin(rad)

            # death: left the map
            if x < min_x or x > max_x or y < min_y or y > max_y:
                break

            # death: touched another snake's body. The same loop accumulates a
            # soft cost for near misses, since a path that skims a body at two
            # pixels only survives the horizon by luck.
            hit = False
            for sx, sy, srad in segments:
                limit = srad + my_radius
                dx = x - sx
                dy = y - sy
                d2 = dx * dx + dy * dy
                if d2 < limit * limit:
                    hit = True
                    break
                soft = limit + clearance
                if d2 < soft * soft:
                    body_penalty += (soft - math.sqrt(d2)) / soft
            if hit:
                break

            # death: ran into a projected opponent head
            for hx, hy, vx, vy, hrad, _hlen in heads:
                px = hx + vx * step
                py = hy + vy * step
                limit = hrad + my_radius
                dx = x - px
                dy = y - py
                if dx * dx + dy * dy < limit * limit:
                    hit = True
                    break
            if hit:
                break

            frames_survived += 1

            # soft cost: crowding a snake that can kill us
            for hx, hy, vx, vy, hrad, _hlen in heads:
                px = hx + vx * step
                py = hy + vy * step
                d = math.hypot(x - px, y - py)
                safe = hrad + my_radius + clearance
                if d < safe:
                    threat_penalty += (safe - d) / safe

            # soft cost: hugging the wall, quadratic so the last few pixels hurt
            # far more than the first ones
            edge = min(x - min_x, max_x - x, y - min_y, max_y - y)
            if edge < wall_range:
                wall_penalty += ((wall_range - edge) / wall_range) ** 2

            # reward: food swept up (matches the environment's eat condition)
            for idx, (fx, fy, size, _d) in enumerate(food):
                if idx in eaten:
                    continue
                if math.hypot(x - fx, y - fy) < my_radius:
                    eaten.add(idx)
                    food_value += size

        # Risk aversion scales with length: an extra point of food is worth
        # proportionally less to a long snake, while a collision costs it more.
        size_ratio = max(1.0, snake.length / self.game.min_length)
        food_scale = 1.0 / math.sqrt(size_ratio)
        caution_scale = math.sqrt(size_ratio)

        score = frames_survived * self.survive_weight
        score += food_value * self.food_weight * food_scale
        score -= threat_penalty * self.threat_weight * caution_scale
        score -= body_penalty * self.body_weight * caution_scale
        score -= wall_penalty * self.wall_weight

        # reward closing on the chosen target, but only if we lived to do it
        if target is not None and frames_survived == self.horizon:
            start_d = distance((snake.x, snake.y), target)
            end_d = distance((x, y), target)
            score += (start_d - end_d) * self.progress_weight

        return score

    # -- decision -----------------------------------------------------------

    def decide(self, snake, view):
        # The chosen plan is held for `commit_frames`, so re-searching every
        # single frame is wasted work. Re-planning every few frames keeps ten
        # searching agents inside the 60 FPS budget; at ~2 px/frame the agent
        # moves only a few pixels between searches.
        if self._cached is not None and self.frames % self.replan_interval != 0:
            return self._cached

        head = (snake.x, snake.y)

        # A rollout travels at most horizon * speed pixels, so anything further
        # away than that (plus clearance) cannot possibly be hit and does not
        # need to be scanned.
        reach = self.horizon * max(0.5, snake.speed) + snake.segment_width + 60.0
        segments, heads = self.nearby_threats(head, radius=reach)
        rollout_food = self.nearby_food(head, radius=reach)

        # Target selection looks further afield than the rollout does, so the
        # agent can steer toward food it will not reach within the horizon.
        target = self.choose_target(snake, self.nearby_food(head))
        food = rollout_food

        best_plan = (TURN_STRAIGHT, TURN_STRAIGHT)
        best_score = -float('inf')

        for first in TURNS:
            for second in TURNS:
                score = self.rollout(snake, (first, second), segments, heads, food, target)
                if score > best_score:
                    best_score = score
                    best_plan = (first, second)

        sprint = self.should_sprint(snake, best_score, target, head, heads)
        self._cached = turn_to_inputs(best_plan[0], sprint)
        return self._cached

    def should_sprint(self, snake, best_score, target, head, heads):
        """
        Sprinting trades one length for speed, so only spend it when the plan is
        clearly safe and there is real distance to close on a target.
        """
        if snake.length < self.game.min_length + self.sprint_min_length_margin:
            return False
        if self.frames - self._last_sprint < self.sprint_cooldown:
            return False
        # never sprint while anything dangerous is nearby
        for hx, hy, _vx, _vy, _hrad, hlen in heads:
            if hlen >= snake.length and distance(head, (hx, hy)) < 220.0:
                return False
        if best_score < self.horizon * self.survive_weight * 0.9:
            return False
        if target is None or distance(head, target) < 120.0:
            return False
        self._last_sprint = self.frames
        return True


class Aggressive_bot(Lookahead_bot):
    """
    Variant for the evaluation write-up. Identical search core, but when we are
    substantially longer than a nearby opponent it re-targets that opponent's
    projected head position instead of food, attempting to cut it off. Included
    to test whether aggression pays in this ruleset, given that contact is fatal
    to whoever touches whom -- a good result to report either way.
    """

    name = "aggressive"
    hunt_length_margin = 25.0
    hunt_radius = 300.0

    def choose_target(self, snake, food):
        head = (snake.x, snake.y)
        prey = None
        prey_dist = float('inf')

        for i, other in enumerate(self.state.snake_list):
            if not other or i == self.player_id:
                continue
            if other.length > snake.length - self.hunt_length_margin:
                continue
            d = distance(head, (other.x, other.y))
            if d < prey_dist and d < self.hunt_radius:
                prey_dist = d
                prey = other

        if prey is not None:
            # aim ahead of the prey, not at it, to cut off its path
            lead = 12.0
            rad = math.radians(prey.heading)
            return (prey.x + prey.speed * lead * math.cos(rad),
                    prey.y + prey.speed * lead * math.sin(rad))

        return super().choose_target(snake, food)


# ---------------------------------------------------------------------------
# Optional bridge to the evolutionary CNN agents
# ---------------------------------------------------------------------------

class Neural_bot(SteeringBot):
    """
    Adapter so an AgentCNN from Pytorch_models can be dropped into the same
    interface: it consumes the 32x24 `view` produced by generate_simple_frame and
    returns the three booleans directly.

    Assign a network before use, e.g.

        bot = Neural_bot(game, 0)
        bot.model = pop_manager.get_agent(idx)

    Falls back to the search agent's behaviour if no model has been attached, so
    the game still runs when torch is unavailable.
    """

    name = "neural"
    model = None

    def __init__(self, game, player_id, model=None):
        super().__init__(game, player_id)
        self.model = model
        self._fallback = Lookahead_bot(game, player_id)

    def decide(self, snake, view):
        if self.model is None or view is None:
            self._fallback.state = self.state
            self._fallback.frames = self.frames
            return self._fallback.decide(snake, view)

        left, right, up = self.model.predict(view)
        return {'left': bool(left), 'right': bool(right), 'up': bool(up)}


AGENT_REGISTRY = {
    # placeholder / baselines
    "circle": Circle_bot,
    "wall-avoid": Wall_bot,
    "greedy": Greedy_bot,
    # potential-field / grid-search family
    "reflex": Reflex_bot,
    "astar": AStar_bot,
    "hunter": Hunter_bot,
    # forward-simulation family
    "reactive": Reactive_bot,
    "lookahead": Lookahead_bot,
    "aggressive": Aggressive_bot,
}
