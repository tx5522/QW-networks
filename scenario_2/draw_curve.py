"""
Copyright 2025 Jiameng Tian and Xing Tang

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the LICENSE file for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env import GridWorld
from classical_agent import BasicPSAgent
from optical_agent import OpticalAgent

# Environment configuration
dimensions = [5, 3]
invalid_cells = {(1, 1), (3, 0)}
rewards = {(4, 0): 10.0}
start_position = (0, 0)

# Training parameters
num_agents = 80
num_trials = 40
max_steps_per_trial = 1000
initial_epsilon = 1.0
epsilon_decay = 0.5

# Used to store the learning curves of all agents
all_curves = np.zeros((num_agents, num_trials), dtype=np.float32)

# Main loop: for each agent
for a_idx in range(num_agents):
    # —— 1. Create environment and agent ——
    env = GridWorld(
        dimensions=dimensions,
        start_position=start_position,
        invalid_cells=invalid_cells,
        rewards=rewards,
    )
    # Choose the agent
    agent = BasicPSAgent(
        num_actions=env.num_actions,
        num_percepts_list=dimensions,
        gamma_damping=0, # Disable forgetting mechanism.
        eta_glow_damping=0.1,
    )
    # agent = OpticalAgent(
    #     num_actions=env.num_actions,
    #     num_percepts_list=dimensions,
    #     eta_glow_damping=0.1,
    #     dim=8,
    #     num_layers=4,
    #     encoding_scheme="superposition",
    # )

    # —— 2. Train one agent ——
    epsilon = initial_epsilon
    learning_curve = np.zeros(num_trials, dtype=np.float32)

    for t in range(num_trials):
        step_count = 0
        observation = env.reset()
        done = False

        while not done and step_count < max_steps_per_trial:
            action = agent.select_action(observation, epsilon)
            observation, reward, done = env.move(action)
            agent.learn_from_reward(reward)
            step_count += 1

        if not done:
            # Force termination if max steps exceeded
            print(
                f"[Agent {a_idx}] Trial {t}: Exceeded maximum steps, forcibly terminated"
            )
        agent.reset_glow()

        learning_curve[t] = step_count
        epsilon *= epsilon_decay

    all_curves[a_idx, :] = learning_curve
    print(f"Finished training agent {a_idx+1}")

# —— 3. Compute mean and standard deviation, then save ——
mean_curve = all_curves.mean(axis=0)
std_curve = all_curves.std(axis=0)

# Create a DataFrame: each column is one agent's learning curve
df_agents = pd.DataFrame(all_curves.T)  # Rows: trials, Columns: agents
df_agents.index.name = "Trial"

# Add columns for mean and standard deviation
df_agents["Mean"] = mean_curve
df_agents["Std"] = std_curve

df_agents.columns = [f"Agent_{i}" for i in range(num_agents)] + ["Mean", "Std"]

# Save as Excel file
excel_filename = "learning_curves.xlsx"
df_agents.to_excel(excel_filename, index=True)

print(f"Learning curve results saved to {excel_filename}")

# —— 4. Plotting ——
trials = np.arange(1, num_trials + 1)

plt.figure(figsize=(10, 6))
plt.plot(trials, mean_curve, label="Mean steps per trial")
plt.fill_between(
    trials,
    mean_curve - std_curve,
    mean_curve + std_curve,
    alpha=0.3,
    label="±1 std. dev.",
)
plt.xlabel("Trial")
plt.ylabel("Steps to reach goal")
plt.title(f"Learning Curve over {num_agents} Agents")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
