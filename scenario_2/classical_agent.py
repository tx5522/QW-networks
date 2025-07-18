"""
This file is adapted from: https://github.com/qic-ibk/projectivesimulation

Original Code Copyright 2018 Alexey Melnikov and Katja Ried.

Modifications copyright (c) 2025 Jiameng Tian and Xing Tang
"""

import numpy as np
from typing import List


class BasicPSAgent:
    """
    Projective Simulation agent with two-layered network.
    """

    def __init__(
        self,
        num_actions: int,
        num_percepts_list: List[int],
        gamma_damping: float,
        eta_glow_damping: float,
    ):
        """
        Initialize the agent.

        Args:
            num_actions: Number of possible actions (>=1).
            num_percepts_list: Cardinality of each perceptual feature (list of ints).
            gamma_damping: Forgetting rate for h-values (0 <= gamma <= 1).
            eta_glow_damping: Damping rate for glow (1 disables glow).
        """
        self.num_actions = num_actions
        self.num_percepts_list = num_percepts_list
        self.gamma_damping = gamma_damping
        self.eta_glow_damping = eta_glow_damping

        self.num_percepts = int(np.prod(num_percepts_list))
        self.h_matrix = np.ones((num_actions, self.num_percepts), dtype=np.float64)
        self.g_matrix = np.zeros((num_actions, self.num_percepts), dtype=np.float64)

    def _encode_percept(self, observation: List[int]) -> int:
        """
        Converts a multi-dimensional percept into a unique integer index.
        """
        index = 0
        multiplier = 1
        for feature_value, cardinality in zip(observation, self.num_percepts_list):
            index += feature_value * multiplier
            multiplier *= cardinality
        return index

    def _update_h_matrix(self, reward: float):
        """
        Applies forgetting and learning using the glow matrix and reward.
        """
        self.h_matrix -= self.gamma_damping * (self.h_matrix - 1.0)
        self.h_matrix += self.g_matrix * reward

    def _update_glow_matrix(self, action: int, percept: int):
        """
        Decays the glow matrix and activates glow for the chosen action-percept pair.
        """
        self.g_matrix *= 1.0 - self.eta_glow_damping
        self.g_matrix[action, percept] = 1.0

    def _compute_action_probabilities(self, percept: int) -> np.ndarray:
        """
        Given a percept index, this method returns a probability distribution over actions.
        """
        h_vector = self.h_matrix[:, percept]
        probabilities = h_vector / np.sum(h_vector)
        return probabilities

    def select_action(self, observation: List[int], epsilon: float) -> int:
        """
        Select an action based on the current percept, and update glow.
        """
        percept = self._encode_percept(observation)
        if np.random.rand() < epsilon:
            action = np.random.randint(self.num_actions)
        else:
            probabilities = self._compute_action_probabilities(percept)
            action = np.random.choice(self.num_actions, p=probabilities)
        self._update_glow_matrix(action, percept)
        return action

    def learn_from_reward(self, reward: float):
        """
        Update h_matrix based on the received reward and current glow matrix.
        """
        self._update_h_matrix(reward)

    def reset_glow(self):
        """
        Reset the g_matrix to 0 at the end of the episode.
        """
        self.g_matrix = np.zeros(
            (self.num_actions, self.num_percepts), dtype=np.float64
        )
