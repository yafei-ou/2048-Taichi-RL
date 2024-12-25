import gymnasium as gym
from gymnasium.wrappers.common import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import os

# Import the custom environment
from env_2048 import Game2048Env

# Configuration parameters
config = {
    "total_timesteps": 1_000_000,  # Total number of training timesteps
    "learning_rate": 0.0003,  # Learning rate for the optimizer
    "n_steps": 2048,  # Number of steps to run for each environment per update
    "batch_size": 64,  # Minibatch size
    "n_epochs": 10,  # Number of epoch when optimizing the surrogate loss
    "gamma": 0.99,  # Discount factor
    "gae_lambda": 0.95,  # GAE lambda
    "clip_range": 0.2,  # Clipping parameter for PPO
    "ent_coef": 0.01,  # Entropy coefficient
    "vf_coef": 0.5,  # Value function coefficient
    "max_grad_norm": 0.5,  # Maximum norm for gradient clipping
    "verbose": 1,  # Verbosity level
    "save_path": "./ppo_2048_model",  # Path to save the trained model
    "eval_freq": 10_000,  # Frequency of evaluations during training
    "log_path": "./logs"  # Path for TensorBoard logs
}

# Create directories if they do not exist
os.makedirs(config["save_path"], exist_ok=True)
os.makedirs(config["log_path"], exist_ok=True)

# Create the environment
def create_env():
    original_env = Game2048Env()
    wrapped_env = TimeLimit(original_env, max_episode_steps=200)
    return wrapped_env

env = DummyVecEnv([create_env])  # Wrapping the custom environment in DummyVecEnv

# Callback for evaluation during training
eval_env = DummyVecEnv([create_env])
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=config["save_path"],
    log_path=config["log_path"],
    eval_freq=config["eval_freq"],
    deterministic=True,
    render=False
)

# Initialize the PPO model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=config["learning_rate"],
    n_steps=config["n_steps"],
    batch_size=config["batch_size"],
    n_epochs=config["n_epochs"],
    gamma=config["gamma"],
    gae_lambda=config["gae_lambda"],
    clip_range=config["clip_range"],
    ent_coef=config["ent_coef"],
    vf_coef=config["vf_coef"],
    max_grad_norm=config["max_grad_norm"],
    verbose=config["verbose"],
    tensorboard_log=config["log_path"]
)

# Train the PPO model
model.learn(total_timesteps=config["total_timesteps"], callback=eval_callback)

# Save the final model
model.save(os.path.join(config["save_path"], "final_model"))

print("Training complete. Model saved at", config["save_path"])
