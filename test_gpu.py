import gymnasium as gym
from gymnasium.wrappers.common import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os

# Import the custom environment
from env_2048 import Game2048Env
from env_2048_vec_gpu import VecTaichiGame2048Env
import taichi as ti
import numpy as np


from collections import defaultdict

def evaluate_env(env: VecTaichiGame2048Env, model, max_episodes: int = 200):
    """
    Evaluates the VecTaichiGame2048Env by running a specified number of episodes
    with the provided model and records the maximum tile value reached.

    :param env: The environment to evaluate.
    :param model: The model used to predict actions.
    :param num_episodes: Number of episodes to run for evaluation.
    :param max_steps_per_episode: Maximum number of steps per episode.
    :return: None
    """

    total_rewards = np.zeros(env.num_envs)
    max_tiles = np.zeros(env.num_envs, dtype=int)  # Track max tiles for each environment
    max_tile_counts = defaultdict(int)  # Count occurrences of specific tiles across all episodes

    obs = env.reset()
    print(f"Initial Observation Shape: {obs.shape}")

    # Track per-environment episode metrics
    step_rewards = np.zeros(env.num_envs)
    completed_episodes = 0

    while completed_episodes < max_episodes:
        # Use the model to predict actions for all environments
        actions, _ = model.predict(obs, deterministic=False)
        env.step_async(actions)
        obs, rewards, dones, infos = env.step_wait()

        # Update maximum tile reached for each environment
        for idx in range(env.num_envs):
            current_max_tile = np.max(obs[idx])
            max_tiles[idx] = max(max_tiles[idx], current_max_tile)

        # Update rewards and completion status
        step_rewards += rewards
        for idx, done in enumerate(dones):
            if done:
                real_max_tile = 2 ** max_tiles[idx]
                max_tile_counts[real_max_tile] += 1  # Increment the max tile count
                print(f"Environment {idx} finished an episode. Total Reward: {step_rewards[idx]}, Max Tile: {real_max_tile}")
                total_rewards[idx] += step_rewards[idx]
                step_rewards[idx] = 0  # Reset step reward for this env after reset

                completed_episodes += 1

    print("\nEvaluation complete!")

    # Print the max tile table
    print("\nMax Tile Table:")
    for tile, count in sorted(max_tile_counts.items(), reverse=True):
        print(f"  Tile {tile}: {count} occurrences")
    
    print(f"Completed episodes: {completed_episodes}")


# Initialize the VecTaichiGame2048Env
ti.init()
num_envs = 1000  # Number of parallel environments
env_max_step = 2000
env = VecTaichiGame2048Env(num_envs=num_envs, grid_size=4, time_limit=env_max_step)

# Model
model = PPO.load("ppo_2048_model/best_model.zip")

evaluate_env(env, model, max_episodes=10_000)

# Close the environment
env.close()
