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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Set, Dict, Optional


class GridWorld:
    """Grid environment with support for rewards, obstacles, and visualization"""

    def __init__(
        self,
        dimensions: List[int],
        start_position: Tuple[int, int] = (0, 0),
        invalid_cells: Optional[Set[Tuple[int, int]]] = None,
        rewards: Optional[Dict[Tuple[int, int], float]] = None,
    ):
        self.grid_size = dimensions
        self.start_position = start_position
        self.position = np.array(start_position)

        self.invalid_cells = set(invalid_cells) if invalid_cells else set()

        self.rewards = np.zeros(dimensions)
        if rewards:
            for (x, y), value in rewards.items():
                self.rewards[x, y] = value
        else:
            self.rewards[-1, -1] = 1.0  # Default goal at bottom-right corner

        self.action_map = {
            0: np.array([0, -1]),  # up
            1: np.array([1, 0]),  # right
            2: np.array([0, 1]),  # down
            3: np.array([-1, 0]),  # left
        }
        self.act_names = ["up", "right", "down", "left"]
        self.num_actions = len(self.action_map)

    def reset(self) -> List[int]:
        """Reset the agent to the starting position"""
        self.position = np.array(self.start_position)
        return self.position.tolist()

    def move(self, action_index: int) -> Tuple[List[int], float, bool]:
        """Perform an action and return new state, reward, and done flag"""
        direction = self.action_map[action_index]
        new_pos = self.position + direction

        if not self._is_valid_position(new_pos):
            return self.position.tolist(), 0.0, False

        self.position = new_pos
        reward = self.rewards[tuple(self.position)].item()
        done = reward != 0.0

        if done:
            self.reset()

        return self.position.tolist(), reward, done

    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check whether the position is within bounds and not an obstacle"""
        x, y = pos
        within_bounds = 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]
        return within_bounds and (x, y) not in self.invalid_cells

    def render(self, show: bool = True):
        """Visualize the current state of the grid"""
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(self.grid_size[0] + 1))
        ax.set_yticks(np.arange(self.grid_size[1] + 1))
        ax.grid(True)

        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                coord = (x, y)
                color = "white"
                if coord in self.invalid_cells:
                    color = "black"
                elif self.rewards[x, y] != 0:
                    color = "red"
                rect = patches.Rectangle(
                    (x, y), 1, 1, facecolor=color, edgecolor="gray"
                )
                ax.add_patch(rect)

        # Draw agent
        ax.add_patch(
            patches.Rectangle(
                tuple(self.position), 1, 1, facecolor="green", edgecolor="gray"
            )
        )

        plt.gca().invert_yaxis()
        plt.title("Grid World")
        if show:
            plt.show()


if __name__ == "__main__":
    invalid_cells = {(1, 1), (2, 2), (3, 3)}
    rewards = {(4, 4): 1.0, (0, 4): -1.0}

    env = GridWorld(
        [5, 5], start_position=(0, 0), invalid_cells=invalid_cells, rewards=rewards
    )
    state = env.reset()
    print("Start at:", state)

    for step in range(30):
        action = np.random.choice(4)
        env.render()
        next_state, reward, done = env.move(action)
        print(
            f"Step {step}: Action {env.act_names[action]}, New State {next_state}, Reward {reward}, Done {done}"
        )
        if done:
            print("Episode finished, resetting.")
