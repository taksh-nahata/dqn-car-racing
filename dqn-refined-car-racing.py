import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99
LR = 5e-5
BATCH_SIZE = 64
BUFFER_SIZE = 15000
TARGET_UPDATE = 2
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.93

class DuelingDQN(nn.Module):
    def __init__(self, action_dim):
        super(DuelingDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

env = gym.make("CarRacing-v3", continuous=False)
action_dim = env.action_space.n

policy_net = DuelingDQN(action_dim).to(device)
target_net = DuelingDQN(action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=BUFFER_SIZE)

epsilon = EPS_START
rewards_history = []

print("Starting REFINED Dueling-DQN Training Loop...")
for episode in range(10): 
    state, _ = env.reset()
    state = np.transpose(state, (2, 0, 1))
    total_reward = 0
    done = False
    
    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                action = policy_net(state_t).max(1)[1].item()
                
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.transpose(next_state, (2, 0, 1))
        done = terminated or truncated
        total_reward += reward
        
        memory.append((state, action, reward, next_state, done))
        state = next_state
        
        if len(memory) > BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states_t = torch.FloatTensor(np.array(states)).to(device) / 255.0
            actions_t = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards_t = torch.FloatTensor(rewards).to(device)
            next_states_t = torch.FloatTensor(np.array(next_states)).to(device) / 255.0
            dones_t = torch.FloatTensor(dones).to(device)
            
            q_values = policy_net(states_t).gather(1, actions_t)
            next_q_values = target_net(next_states_t).max(1)[0].detach()
            expected_q_values = rewards_t + (GAMMA * next_q_values * (1 - dones_t))
            
            loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    rewards_history.append(total_reward)
    print(f"Episode {episode+1} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f}")
    
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

plt.figure(figsize=(10, 5))
plt.plot(rewards_history, marker="o", color="green", label="Refined Dueling DQN")
plt.title("Refined Dueling DQN Learning Progress on CarRacing-v3")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.savefig("dqn_refined_learning_curve.png")
print("Saved dqn_refined_learning_curve.png successfully!")