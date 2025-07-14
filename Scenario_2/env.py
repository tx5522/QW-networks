import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Set, Dict, Optional


class GridWorld:
    """A 2D grid environment supporting rewards, obstacles, and visualization."""

    def __init__(
        self,
        dimensions: Tuple[int, int],
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

    def reset(self) -> np.ndarray:
        """Reset the agent to the starting position."""
        self.position = np.array(self.start_position)
        return self.position

    def move(self, action_index: int) -> Tuple[np.ndarray, float, bool]:
        """Execute an action and return the new state, reward, and completion flag."""
        direction = self.action_map[action_index]
        new_pos = self.position + direction

        if not self._is_valid_position(new_pos):
            return self.position, 0.0, False

        self.position = new_pos
        reward = self.rewards[tuple(self.position)]
        done = reward != 0.0

        if done:
            self.reset()

        return self.position, reward, done

    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if a position is valid (within bounds and not an obstacle)."""
        x, y = pos
        within_bounds = 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]
        return within_bounds and (x, y) not in self.invalid_cells

    def render(self, show: bool = True):
        """Visualize the current state."""
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

        # Draw the agent
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
        (5, 5), start_position=(0, 0), invalid_cells=invalid_cells, rewards=rewards
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