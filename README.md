# Deep Q-Network (DQN) for Autonomous Driving Simulation

This repository contains a custom, scratch-built implementation of the Deep Q-Network (DQN) algorithm applied to the `CarRacing-v3` visual simulation environment. Developed and deployed on the NRP Nautilus distributed Kubernetes cluster, this project serves as a technical exploration of value-based deep reinforcement learning comparing a standard baseline DQN with a refined Dueling DQN architecture.

## 🧠 Algorithm Mechanics (PyTorch)
Unlike actor-critic methods, these implementations focus strictly on value-based action selection using:
*   **Deep Q-Network (Baseline):** A standard Deep Q-Network (CNN + Linear Layers) that directly estimates action-values ($Q$-values) from raw visual inputs.
*   **Dueling DQN (Refined):** An architectural upgrade that splits the network's head into a state-value stream ($V(s)$) and an action-advantage stream ($A(s, a)$) to stabilize learning.
*   **Experience Replay Buffer:** A rolling memory bank that breaks sequential frame correlation by randomly sampling past transitions.
*   **Target Network:** A secondary, periodically-frozen cloned network to stabilize value target calculations.
*   **Epsilon-Greedy Exploration:** Gradually transitions the agent from exploration (random actions) to exploitation (policy actions) as training episodes progress.

## 📂 Repository Structure
*   `dqn-car-racing.py`: Original baseline DQN algorithm training execution script.
*   `dqn-refined-car-racing.py`: Refined Dueling DQN algorithm training execution script.
*   `dqn_learning_curve.png`: Generated training reward visualization for the baseline model.
*   `dqn_refined_learning_curve.png`: Generated training reward visualization for the refined Dueling model.
*   `Deep Q-Networks (DQN).pptx`: Overview presentation explaining the core mathematical mechanics of DQN.

## 💻 How to Run (NRP Nautilus / Kubernetes Pod)
To execute the training loops headlessly inside a remote Linux container, initialize the virtual frame buffer (`xvfb`):

```bash
# Install dependencies
pip install gymnasium[box2d] pygame torch matplotlib

# Run Baseline DQN Training
xvfb-run -s "-screen 0 1024x768x24" python dqn-car-racing.py

# Run Refined Dueling DQN Training
xvfb-run -s "-screen 0 1024x768x24" python dqn-refined-car-racing.py