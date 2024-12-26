import gymnasium as gym
from gymnasium.wrappers.common import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import os

# Import the custom environment
from env_2048 import Game2048Env


# Create the environment
def create_env():
    original_env = Game2048Env()
    wrapped_env = TimeLimit(original_env, max_episode_steps=500)
    return wrapped_env

# Model
model = PPO.load("ppo_2048_model/best_model.zip")

# Optional: Test the trained model
env = create_env()
obs, _ = env.reset()
terminated = False
truncated = False
while not terminated and not truncated:
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward)
    env.render()
print(env.env.score)