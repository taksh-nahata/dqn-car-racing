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
BATCH_SIZE = 64
EPISODES = 10

class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

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

def train_agent(algo_name, model_class, lr, buffer_size, target_update, eps_decay):
    print(f"\\\\n--- Training {algo_name} ---")
    env = gym.make("CarRacing-v3", continuous=False)
    action_dim = env.action_space.n
    
    policy_net = model_class(action_dim).to(device)
    target_net = model_class(action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = deque(maxlen=buffer_size)
    
    epsilon = 1.0
    rewards_history = []
    losses_history = []
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        state = np.transpose(state, (2, 0, 1))
        total_reward = 0
        episode_losses = []
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
                
                episode_losses.append(loss.item())
                
        epsilon = max(0.05, epsilon * eps_decay)
        rewards_history.append(total_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        losses_history.append(avg_loss)
        print(f"Episode {episode+1} | Reward: {total_reward:.2f} | Avg Loss: {avg_loss:.4f} | Epsilon: {epsilon:.2f}")
        
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
    env.close()
    return rewards_history, losses_history

dqn_rewards, dqn_losses = train_agent(
    algo_name="Standard DQN Baseline",
    model_class=DQN,
    lr=1e-4,
    buffer_size=10000,
    target_update=5,
    eps_decay=0.90
)

dueling_rewards, dueling_losses = train_agent(
    algo_name="Refined Dueling DQN",
    model_class=DuelingDQN,
    lr=5e-5,
    buffer_size=15000,
    target_update=2,
    eps_decay=0.93
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(dqn_rewards, marker="o", color="red", label="Standard DQN (Baseline)")
ax1.plot(dueling_rewards, marker="o", color="green", label="Dueling DQN (Refined)")
ax1.set_title("Benchmarked Episodic Rewards")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.grid(True)
ax1.legend()

ax2.plot(dqn_losses, marker="s", color="orange", label="Standard DQN (Baseline)")
ax2.plot(dueling_losses, marker="s", color="blue", label="Dueling DQN (Refined)")
ax2.set_title("Benchmarked Training Loss Progression")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Mean MSE Loss")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig("benchmark_results.png")
print("\\\\nSaved benchmark_results.png successfully!")