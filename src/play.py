import os
import time
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from gymnasium.wrappers import FrameStack, AtariPreprocessing
from huggingface_sb3 import load_from_hub
from datetime import datetime

# Generate a random seed
SEED = random.randint(0, 10000)

# Load the model from Hugging Face Hub
checkpoint = load_from_hub("ThomasSimonini/ppo-PongNoFrameskip-v4", "ppo-PongNoFrameskip-v4.zip")

# Handle any custom objects due to Pickle version mismatch
custom_objects = {
    "learning_rate": 1e-4,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}
# Load the PPO model
model = PPO.load(checkpoint, custom_objects=custom_objects)

# Set up the Atari environment
def make_env():
    env = gym.make('PongNoFrameskip-v4', frameskip=1, render_mode="human")
    env.action_space.seed(SEED)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4, scale_obs=False)
    env = FrameStack(env, num_stack=4)
    return env

# Init Pong environment
env = make_env()
time.sleep(1e-2)
log_file = "src/performance_log.txt"
os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Check if the directory exists

episode_rewards = []
episode_lengths = []

# Main loop
with open(log_file, "a") as log:
    for episode in range(100):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0  # Initialize step counter
        done = False

        start_time = datetime.now()  # Start time for the episode

        while not done:
            action, _states = model.predict(np.array(obs), deterministic=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
            time.sleep(0.001)

            total_reward += reward
            steps += 1  # Increment step counter
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)  # Store the episode length

        end_time = datetime.now()
        episode_duration = end_time - start_time
        log.write(f"Episode {episode + 1}: Total Reward = {total_reward}, Duration = {episode_duration}, Steps = {steps}\n")
        log.flush()
        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Duration = {episode_duration}, Steps = {steps}")

env.close()

# Calculate average metrics
average_reward = np.mean(episode_rewards)
average_length = np.mean(episode_lengths)
print(f"Average Reward: {average_reward}")
print(f"Average Episode Length: {average_length}")

# Plot the results for episode rewards
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(episode_rewards, label="Episode Reward", color="red")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Pong Agent Performance - Episode Rewards")
plt.legend()
plt.grid()

# Plot the results for episode lengths
plt.subplot(1, 2, 2)
plt.plot(episode_lengths, label="Episode Length", color="orange")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Pong Agent Performance - Episode Lengths")
plt.legend()
plt.grid()

# Show the plots
plt.suptitle("Pong Agent Key Performance Indicators")
plt.tight_layout()
plt.show()

