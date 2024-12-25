import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        
        self.grid_size = 4
        self.action_space = spaces.Discrete(4)  # Actions: 0 = Up, 1 = Down, 2 = Left, 3 = Right
        self.observation_space = spaces.Box(low=0, high=16, shape=(self.grid_size, self.grid_size), dtype=np.int32)
        
        self.grid = None
        self.score = 0
        self.reward = 0
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the game to the initial state."""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self._spawn_block()
        self._spawn_block()
        self.score = 0
        return self.grid, None

    def step(self, action):
        """Execute one time step within the environment."""
        assert self.action_space.contains(action), "Invalid Action"
        
        grid_before = self.grid.copy()

        self.reward = 0
        
        if action == 0:  # Up
            self._move_up()
        elif action == 1:  # Down
            self._move_down()
        elif action == 2:  # Left
            self._move_left()
        elif action == 3:  # Right
            self._move_right()

        if not np.array_equal(grid_before, self.grid):
            self._spawn_block()

        done = self._is_game_over()
        reward = self.reward
        
        return self.grid, reward, done, False, {}

    def render(self):
        """Render the current state of the game."""
        for row in self.grid:
            print("\t".join(str(2**x if x > 0 else 0) for x in row))
        print()

    def _move_up(self):
        self.grid = np.transpose(self.grid)
        self._move_left()
        self.grid = np.transpose(self.grid)

    def _move_down(self):
        self.grid = np.transpose(self.grid)
        self._move_right()
        self.grid = np.transpose(self.grid)

    def _move_left(self):
        for i in range(self.grid_size):
            self.grid[i] = self._merge(self.grid[i])

    def _move_right(self):
        for i in range(self.grid_size):
            self.grid[i] = self._merge(self.grid[i][::-1])[::-1]

    def _merge(self, row):
        """Merge a single row or column."""
        non_zero = row[row != 0]
        new_row = []
        skip = False

        for j in range(len(non_zero)):
            if skip:
                skip = False
                continue

            if j + 1 < len(non_zero) and non_zero[j] == non_zero[j + 1]:
                new_row.append(non_zero[j] + 1)
                self.score += 2 ** (non_zero[j] + 1)
                self.reward += non_zero[j] - 1 # 4-->0
                skip = True
            else:
                new_row.append(non_zero[j])

        while len(new_row) < self.grid_size:
            new_row.append(0)

        return np.array(new_row, dtype=np.int32)

    def _spawn_block(self):
        """Spawn a new block (2 or 4) at a random empty position."""
        empty_positions = list(zip(*np.where(self.grid == 0)))
        if empty_positions:
            x, y = random.choice(empty_positions)
            self.grid[x, y] = 1 if random.random() < 0.9 else 2  # 1 for 2, 2 for 4

    def _is_game_over(self):
        """Check if the game is over."""
        if np.any(self.grid == 0):
            return False

        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                if self.grid[i, j] == self.grid[i, j + 1] or self.grid[j, i] == self.grid[j + 1, i]:
                    return False

        return True

# Registering the environment
if __name__ == "__main__":
    env = Game2048Env()
    env.reset()
    env.render()
    done = False

    while not done:
        action = int(input("Enter action (0: Up, 1: Down, 2: Left, 3: Right): "))
        state, reward, done, info = env.step(action)
        env.render()
        print(f"Score: {reward}")

    print("Game Over")
