# Deep Q-Network (DQN) for Autonomous Driving Simulation

This repository contains a custom, scratch-built implementation of the Deep Q-Network (DQN) algorithm applied to the `CarRacing-v3` visual simulation environment. Developed and deployed on the NRP Nautilus distributed Kubernetes cluster, this project serves as a technical exploration of off-policy, value-based deep reinforcement learning.

## 🧠 Algorithm Mechanics (PyTorch)
Unlike actor-critic methods, this implementation focuses strictly on value-based action selection using:
*   **Convolutional Neural Network (CNN):** Processes raw visual frame inputs from the game screen to evaluate road geometries.
*   **Experience Replay Buffer:** A rolling memory bank that breaks sequential correlation by randomly sampling past states.
*   **Target Network:** A secondary, periodically-frozen cloned network to stabilize value target calculations.
*   **Epsilon-Greedy Exploration:** Gradually transitions the agent from 100% exploration to 95% exploitation as the training epochs progress.

## 📂 Repository Structure
*   `dqn_car_racing.py`: Custom DQN algorithm training execution script.
*   `dqn_learning_curve.png`: Generated training reward visualization plot.

## 💻 How to Run (NRP Nautilus / Kubernetes Pod)
To execute the training loop headlessly inside a remote Linux container, initialize the virtual frame buffer (`xvfb`):

```bash
# Install dependencies
pip install gymnasium[box2d] pygame torch matplotlib

# Run DQN Training
xvfb-run -s "-screen 0 1024x768x24" python dqn_car_racing.py