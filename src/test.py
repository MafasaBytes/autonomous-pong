
#import gymnasium as gym
# print(gym.envs.registry.keys())

# import ale_py
# print(ale_py.roms.list_roms())

# import gymnasium as gym

# try:
#     env = gym.make("ALE/Pong-v5")
#     env.reset()
#     print("Environment loaded successfully.")
# except Exception as e:
#     print(f"Failed to load environment: {e}")

import torch

# For PyTorch models
# try:
#     state_dict = torch.load("models/pong_dqn_model.pth", map_location=torch.device('cpu'))
#     print("Model loaded successfully:", state_dict.keys())
# except AssertionError as e:
#     print("Error loading model:", e)


# import gymnasium as gym
#
# env = gym.make("ale_py:ALE/Pong-v5")

import gym
import ale_py

# print('gym:', gym.__version__)
# print('ale_py:', ale_py.__version__)
#
# envs = gym.envs.registry.all()
# env_names = [env_spec.id for env_spec in envs]
# print(env_names)

import gym

envs = gym.envs.registry.get('Pong-v4')

if not envs:
    print(f"Pong version {envs}is installed and available.")

