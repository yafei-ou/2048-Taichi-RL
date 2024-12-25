import random
import taichi as ti
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from typing import Any, Optional, Sequence, Union, Iterable


@ti.data_oriented
class VecTaichiGame2048Env(VecEnv):
    """
    A vectorized version of the 2048 environment using Taichi for GPU acceleration.
    It handles multiple 2048 "sub-environments" in parallel.
    """

    def __init__(
        self,
        num_envs: int,
        grid_size: int = 4,
        time_limit: int = 200
    ):
        """
        :param num_envs: number of parallel sub-environments
        :param grid_size: size of the 2048 grid (usually 4)
        """

        # We set the observation and action spaces similarly to the single env
        observation_space = spaces.Box(low=0, high=16, shape=(grid_size, grid_size), dtype=np.int32)
        action_space = spaces.Discrete(4)  # 0=Up, 1=Down, 2=Left, 3=Right

        super().__init__(
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )

        self.grid_size = grid_size
        self.time_limit = time_limit

        # Initialize Taichi
        # If you have a GPU, you can try ti.gpu
        # If you want to fall back on CPU, you can use ti.cpu
        # or just do: ti.init(arch=ti.gpu, debug=False)
        ti.init(arch=ti.gpu, debug=False)

        # ==============
        # Taichi Fields
        # ==============
        # Let's store the grid states for all envs as a 3D field:
        # shape = (num_envs, grid_size, grid_size)
        self.grid = ti.field(dtype=ti.i32, shape=(num_envs, grid_size, grid_size))

        # We also store a done-mask: 0 = not done, 1 = done
        self.done = ti.field(dtype=ti.i32, shape=(num_envs,))
        self.truncated = ti.field(dtype=ti.i32, shape=(num_envs,))

        # We store the reward for each environment on each step
        self.rewards = ti.field(dtype=ti.f32, shape=(num_envs,))

        # We'll store the actions in a buffer to implement step_async / step_wait
        self.actions_buffer = ti.field(dtype=ti.i32, shape=(num_envs,))

        self.step_counts = ti.field(dtype=ti.i32, shape=(num_envs,))

        # For reproducible random spawning, you can store seeds or states here.
        # We'll do something simple for demonstration:
        self.rng_seeds = ti.field(dtype=ti.i32, shape=(num_envs,))
        for i in range(num_envs):
            # Just use some random seeds
            self.rng_seeds[i] = random.randint(0, 2**31 - 1)

        # Initialize the environment states
        self._reset_all_envs()

    # ----------------------------
    #  Core VecEnv API methods
    # ----------------------------

    def reset(self):
        """
        Reset all sub-environments.
        Returns the initial observation [num_envs, grid_size, grid_size]
        """
        self._reset_all_envs()
        obs = self._get_observations()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        """
        Store the actions in Taichi's buffer so we can process them in step_wait().
        """
        # We assume len(actions) == num_envs
        for i in range(self.num_envs):
            self.actions_buffer[i] = actions[i]

    def step_wait(self):
        """
        Execute the step. We run a kernel that:
          1. Resets the rewards
          2. Performs the moves in parallel
          3. Spawns new blocks if a move was valid
          4. Checks for done
        Returns (obs, reward, done, info)
        """
        self._step_kernel()
        obs = self._get_observations()
        r = self._get_rewards()
        d = self._get_done()
        t = self._get_truncated()
        info = [{} for _ in range(self.num_envs)]  # empty info
        for e in range(self.num_envs):
            info[e]["TimeLimit.truncated"] = t[e] and not d[e]
            d[e] = d[e] or t[e]
            if d[e]:
                # save final observation where user can get it, then reset
                info[e]["terminal_observation"] = np.copy(obs[e])
                self._reset_env(e)
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        obs[e, i, j] = self.grid[e, i, j]
        return obs, r, d, info

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        # For simplicity, we’ll just handle known attributes
        if attr_name == "render_mode":
            # Not used in this example
            return [None] * self.num_envs
        else:
            # Return something or raise an error
            return [None] * self.num_envs

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        pass

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list[Any]:
        # Not used extensively here, but typically you'd call a python method
        # for each sub-env. We have everything in kernels though.
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices=None) -> list[bool]:
        # We have no wrappers
        return [False] * self.num_envs

    # ----------------------------
    #  Optional Rendering
    # ----------------------------
    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        """
        If you want to visualize each environment's state as an image,
        you can create an RGB array. Here, we just return None for simplicity.
        """
        return [None for _ in range(self.num_envs)]

    # -----------------------------------
    #  Helper methods + Taichi Kernels
    # -----------------------------------

    def _reset_all_envs(self):
        """
        Reset the entire vector of envs in parallel using a kernel.
        """
        self._reset_kernel()

    @ti.kernel
    def _reset_env(self, e: ti.i32):
        self.step_counts[e] = 0
        self.done[e] = 0
        self.truncated[e] = 0
        # Clear the grid
        for i, j in ti.ndrange(self.grid_size, self.grid_size):
            self.grid[e, i, j] = 0

        # Spawn two blocks
        self._spawn_block_in_env(e)
        self._spawn_block_in_env(e)

    @ti.kernel
    def _reset_kernel(self):
        for e in range(self.num_envs):
            # Mark done = 0
            self.done[e] = 0
            self.truncated[e] = 0

            # Clear the grid
            for i, j in ti.ndrange(self.grid_size, self.grid_size):
                self.grid[e, i, j] = 0

            # Spawn two blocks
            self._spawn_block_in_env(e)
            self._spawn_block_in_env(e)

    def _get_observations(self) -> np.ndarray:
        """
        Copy the grid from GPU to CPU as a numpy array of shape (num_envs, grid_size, grid_size).
        """
        obs = np.zeros((self.num_envs, self.grid_size, self.grid_size), dtype=np.int32)
        for e in range(self.num_envs):
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    obs[e, i, j] = self.grid[e, i, j]
        return obs

    def _get_rewards(self) -> np.ndarray:
        """
        Copy the rewards from GPU to CPU.
        """
        r = np.zeros((self.num_envs,), dtype=np.float32)
        for e in range(self.num_envs):
            r[e] = self.rewards[e]
        return r

    def _get_done(self) -> np.ndarray:
        """
        Copy the done array from GPU to CPU (and convert to bool).
        """
        d = np.zeros((self.num_envs,), dtype=bool)
        for e in range(self.num_envs):
            d[e] = (self.done[e] == 1)
        return d
    
    def _get_truncated(self):
        """
        Copy the truncated array from GPU to CPU (and convert to bool).
        """
        t = np.zeros((self.num_envs,), dtype=bool)
        for e in range(self.num_envs):
            t[e] = (self.truncated[e] == 1)
        return t

    @ti.kernel
    def _step_kernel(self):
        """
        This kernel:
        1. Resets rewards
        2. For each env, if not done -> apply the stored action
        3. If a move changed the grid, spawn block
        4. Check done
        """
        # Step 1: Reset rewards
        for e in range(self.num_envs):
            self.rewards[e] = 0  # the default step penalty

        # Step 2: Execute the move in parallel
            if self.done[e] == 0:
                old_grid_hash = self._grid_hash(e)
                action = self.actions_buffer[e]
                if action == 0:  # Up
                    self._move_up(e)
                elif action == 1:  # Down
                    self._move_down(e)
                elif action == 2:  # Left
                    self._move_left(e)
                elif action == 3:  # Right
                    self._move_right(e)
                
                new_grid_hash = self._grid_hash(e)
                # If the grid changed, spawn a new block
                if old_grid_hash != new_grid_hash:
                    self._spawn_block_in_env(e)

        # Step 3: Check "done" after the move
            if self.done[e] == 0:  # only check if not already done
                if self._is_game_over(e):
                    self.done[e] = 1

            self.step_counts[e] += 1
            if self.step_counts[e] > self.time_limit:
                self.step_counts[e] = 0
                self.truncated[e] = 1

    @ti.func
    def _grid_hash(self, e) -> ti.u32:  # explicitly return u32, if desired
        # Use an unsigned 32-bit local variable
        h = ti.u32(0)
        # Predefine a mask as an unsigned 32-bit literal
        mask = ti.u32(0xFFFFFFFF)
        
        for i, j in ti.ndrange(self.grid_size, self.grid_size):
            # Convert operands to u32 so everything stays in range
            h = (h * ti.u32(31) + (ti.u32(self.grid[e, i, j]) + ti.u32(1))) & mask
        return h

    # -----------
    #  Moves
    # -----------
    @ti.func
    def _move_up(self, e):
        self._transpose(e)
        self._move_left(e)
        self._transpose(e)

    @ti.func
    def _move_down(self, e):
        self._transpose(e)
        self._move_right(e)
        self._transpose(e)

    @ti.func
    def _move_left(self, e):
        for i in range(self.grid_size):
            self._merge_row(e, i, reverse=False)

    @ti.func
    def _move_right(self, e):
        for i in range(self.grid_size):
            self._merge_row(e, i, reverse=True)

    @ti.func
    def _transpose(self, e):
        # Transpose grid[e]
        for i, j in ti.ndrange(self.grid_size, self.grid_size):
            if i < j:
                tmp = self.grid[e, i, j]
                self.grid[e, i, j] = self.grid[e, j, i]
                self.grid[e, j, i] = tmp

    @ti.func
    def _merge_row(self, e, row_idx, reverse):
        # return
        """
        Merge logic for one row.
        `reverse=False` for left, True for right.
        """

        row = ti.static(self.grid_size)
        # read row
        tmp = ti.Vector([0, 0, 0, 0])
        compressed = ti.Vector([0, 0, 0, 0])
        merged = ti.Vector([0, 0, 0, 0])
        final_row = ti.Vector([0, 0, 0, 0])

        for c in range(row):
            if reverse:
                tmp[c] = self.grid[e, row_idx, row - 1 - c]
            else:
                tmp[c] = self.grid[e, row_idx, c]

        # compress (remove 0s)
        for i in range(row):
            compressed[i] = 0

        idx = 0
        for i in range(row):
            if tmp[i] != 0:
                compressed[idx] = tmp[i]
                idx += 1

        # merge
        for i in range(row):
            merged[i] = 0

        skip = False
        out_idx = 0
        for i in range(row):
            if skip:
                skip = False
                continue

            if i < row - 1 and compressed[i] == compressed[i+1] and compressed[i] != 0:
                # merge
                new_val = compressed[i] + 1
                merged[out_idx] = new_val

                # Reward logic example: add a small positive reward for merges
                # e.g. 2 -> 4 or 4 -> 8, etc.
                # You can scale it or shape it how you want
                # For a tile 'new_val', the actual 2048 tile value is 2^new_val.
                # We’ll just add something simpler:
                self.rewards[e] += new_val - 2

                skip = True
            else:
                merged[out_idx] = compressed[i]
            out_idx += 1

        # re-pack to finalize row
        for i in range(row):
            final_row[i] = 0
        for i in range(out_idx):
            final_row[i] = merged[i]

        # write row back
        for c in range(row):
            if reverse:
                self.grid[e, row_idx, row - 1 - c] = final_row[c]
            else:
                self.grid[e, row_idx, c] = final_row[c]

    # -----------
    #  Spawning
    # -----------
    @ti.func
    def _spawn_block_in_env(self, e):
        """
        Spawn a new block (2 or 4) in a random empty spot, if available.
        We'll do a quick approach using a random search (not super efficient).
        """
        empty_count = 0
        for i, j in ti.ndrange(self.grid_size, self.grid_size):
            if self.grid[e, i, j] == 0:
                empty_count += 1

        if empty_count != 0:
            # pick a random empty index in [0..empty_count-1]
            # We rely on taichi's built-in random for demonstration.
            # If you want stable reproducibility, you can use self.rng_seeds[e].
            r = int(ti.random() * empty_count)

            idx = 0
            chosen_i = 0
            chosen_j = 0
            for i, j in ti.ndrange(self.grid_size, self.grid_size):
                if self.grid[e, i, j] == 0:
                    if idx == r:
                        chosen_i = i
                        chosen_j = j
                        break
                    idx += 1

            # 90% chance 2, 10% chance 4 -> In our notation: tile=1 => actual=2, tile=2 => actual=4
            if ti.random() < 0.9:
                self.grid[e, chosen_i, chosen_j] = 1
            else:
                self.grid[e, chosen_i, chosen_j] = 2

    # -----------
    #  Game over?
    # -----------
    @ti.func
    def _is_game_over(self, e) -> ti.i32:
        """
        Return 1 if game is over, else 0
        """
        done = True
        # If there's any empty cell => not done
        for i, j in ti.ndrange(self.grid_size, self.grid_size):
            if self.grid[e, i, j] == 0:
                done = False

        # Check merges
        for i, j in ti.ndrange(self.grid_size, self.grid_size):
            if j < self.grid_size - 1:
                if self.grid[e, i, j] == self.grid[e, i, j+1]:
                    done = False
            if i < self.grid_size - 1:
                if self.grid[e, i, j] == self.grid[e, i+1, j]:
                    done = False

        return done


if __name__ == "__main__":
    # Create a vectorized environment with 8 parallel 2048 boards
    vec_env = VecTaichiGame2048Env(num_envs=2, grid_size=4, time_limit=3)

    obs = vec_env.reset()
    print("Initial observation shape:", obs.shape)  # (8, 4, 4)

    # Let’s do a random rollout for a few steps
    for step in range(5):
        actions = np.random.randint(0, 4, size=(2,))
        vec_env.step_async(actions)
        obs, rewards, dones, infos = vec_env.step_wait()

        print("Step:", step)
        print("Obs shape:", obs.shape)
        print("Obs:", obs)
        print("Infos:", infos)
        print("Rewards:", rewards)
        print("Dones:", dones)

        # if all(dones):
        #     # If all envs done, reset them
        #     obs = vec_env.reset()
        #     print("All envs done, resetting...")

    vec_env.close()
