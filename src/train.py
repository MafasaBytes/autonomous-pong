import gym
import torch
import torch.optim as optim
import numpy as np
import cv2
from collections import deque
from replay_memory import ReplayMemory
from model import DQN_Agent
import matplotlib.pyplot as plt

np.bool = np.bool_

def flatten_and_convert_to_float(data):
    flat_list = []
    if isinstance(data, dict):
        data = list(data.values())
    for item in data:
        if isinstance(item, (list, tuple, np.ndarray)):
            flat_list.extend(flatten_and_convert_to_float(item))
        else:
            try:
                flat_list.append(float(item))
            except (TypeError, ValueError):
                print(f"Skipping non-numeric value: {data}")
    return np.array(flat_list, dtype=np.float32)

def train_agent(env, model, episodes=1, batch_size=32, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    memory = ReplayMemory(50000)
    target_net = DQN_Agent(env.action_space.n)
    target_net.load_state_dict(model.state_dict())
    target_net.eval()

    episode_rewards = []
    epsilon_values = []
    losses = []

    print("Starting training...")

    for e in range(episodes):
        print(f"Starting episode {e}...")
        state = env.reset()

        # Preprocess the initial state
        state_numeric = flatten_and_convert_to_float(state)
        state_numeric = cv2.resize(state_numeric.reshape(210, 160, 3).mean(axis=2), (84, 84)) / 255.0
        state_stack = deque([state_numeric] * 4, maxlen=4)
        total_reward = 0
        done = False
        step = 0

        while not done:
            # Convert frame stack to a tensor of shape [4, 84, 84]
            state_tensor = torch.FloatTensor(np.array(state_stack).reshape(4, 84, 84)).unsqueeze(0)

            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = model(state_tensor).argmax().item()

            # Take action and observe the result
            next_state, reward, terminated, truncated, _ = env.step(action)
            reward = np.clip(reward, -1, 1)  # Optional reward clipping
            total_reward += reward
            step += 1
            done = terminated or truncated

            # Process next_state
            next_state_numeric = flatten_and_convert_to_float(next_state)
            next_state_numeric = cv2.resize(next_state_numeric.reshape(210, 160, 3).mean(axis=2), (84, 84)) / 255.0
            state_stack.append(next_state_numeric)

            # Store the transition in memory
            memory.push(np.array(state_stack), action, reward, np.array(state_stack), done)

            # Perform optimization if memory has enough samples
            if len(memory) > batch_size:
                # Sample a batch from the replay memory
                batch = memory.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.FloatTensor(np.array(states)).reshape(batch_size, 4, 84, 84)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(np.array(next_states)).reshape(batch_size, 4, 84, 84)
                dones = torch.FloatTensor(dones)

                # Compute Q-values
                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0].detach()
                expected_q_values = rewards + gamma * next_q_values * (1 - dones)

                # Compute loss
                loss = torch.nn.functional.mse_loss(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                print(f"Episode {e} Step {step}, Loss: {loss.item():.4f}")

        # Append total reward and epsilon
        episode_rewards.append(total_reward)
        epsilon_values.append(epsilon)

        # Reduce epsilon-greedy decay
        epsilon = max(epsilon * epsilon_decay, 0.01)
        if e % 5 == 0:
            target_net.load_state_dict(model.state_dict())

        print(f"Episode {e} complete - Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    print("Training complete!")

    # Save model
    torch.save(model.state_dict(), 'models/pong_dqn_model_fine_tuned.pth')
    print("Model saved as pong_dqn_model_fine_tuned.pth")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(epsilon_values)
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay over Episodes')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(losses)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss per Step')
    plt.show()

# Main code execution
if __name__ == "__main__":
    # Initialize environment and model
    env = gym.make("PongNoFrameskip-v4")
    model = DQN_Agent(env.action_space.n)
    
    # Start training
    train_agent(env, model)
