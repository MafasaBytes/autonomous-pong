from train import train_agent
from model import DQN_Agent
import gym
import torch
import os

MODEL_PATH = "/Users/mafasa/Desktop/MSc/Reinforcement Learning/pong_rl_project/rl_env/models/pong_dqn_model.pth"

def main():
    env = gym.make('Pong-v4')
    action_size = env.action_space.n
    model = DQN_Agent(action_size)

    if not os.path.exists(MODEL_PATH):
        print(f"Loading DQN_model")
        model.load_state_dict(torch.load(MODEL_PATH))

        raise TypeError("No such file or directory")  


if __name__ == '__main__':
    main()
