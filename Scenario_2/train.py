import numpy as np
import matplotlib.pyplot as plt
from env import GridWorld
from classical_agent import BasicPSAgent
from optical_agent import OpticalAgent

# Build Grid World environment
dimensions = (9, 6)
invalid_cells = {(2, 1), (2, 2), (2, 3), (5, 4), (7, 0), (7, 1), (7, 2)}
rewards = {
    (8, 0): 100,  # Goal position with positive reward
}
start_position = (0, 2)  # Agent's starting position

env = GridWorld(
    dimensions=dimensions,
    start_position=start_position,
    invalid_cells=invalid_cells,  # Obstacle positions
    rewards=rewards,
)
env.render()  # Visualize the initial environment

# Initialize classical agent
agent = BasicPSAgent(
    num_actions=env.num_actions,
    num_percepts_list=dimensions,  # Perceptual space dimensions
    gamma_damping=0,
    eta_glow_damping=0.07,  # Glow decay rate
    policy_type="standary",
    beta_softmax=1,  # Softmax temperature
)
# Initialize optical agent (alternative)
# agent = OpticalAgent(
#     num_actions=env.num_actions,
#     num_percepts_list=dimensions,
#     eta_glow_damping=0.07,
#     dim=18,
#     num_layers=18,
#     encoding_scheme="superposition",
# )

num_trials = 100  # Total number of training trials
max_steps_per_trial = 5000  # Maximum steps allowed per trial

learning_curve = np.zeros(num_trials)  # Record steps per trial
epsilon = 1  # Initial exploration rate (ε-greedy)

# Start training
for i_trial in range(num_trials):
    step_count = 0
    observation = env.reset()  # Reset environment to initial state
    done = False

    while not done:
        # Select action using ε-greedy strategy
        action = agent.select_action(observation, epsilon)
        # Execute action in environment
        new_observation, reward, done = env.move(action)
        observation = new_observation
        # Update agent with reward
        agent.learn_from_reward(reward)

        step_count += 1
        # Force termination if exceeding maximum steps
        if step_count > max_steps_per_trial:
            print(f"Trial {i_trial}: Exceeded maximum step limit, forced termination")
            break
    
    agent.reset_glow()  # Clear glow memory after each trial

    # Decay exploration rate (ε)
    epsilon *= 0.6

    learning_curve[i_trial] = step_count

    # Print progress every n trials
    n = 3
    if (i_trial + 1) % n == 0:
        # Calculate average steps over recent n trials
        avg_steps = np.mean(learning_curve[max(0, i_trial - n + 1) : (i_trial + 1)])
        print(f"Trial {i_trial+1}: Average steps over last {n} trials = {avg_steps:.2f}")

# Plot learning curve
plt.figure(figsize=(10, 5))
plt.plot(learning_curve, label="Steps per trial")
plt.xlabel("Trial")
plt.ylabel("Steps to reach goal")
plt.title("Agent Learning Curve")
plt.grid(True)
plt.legend()
plt.show()