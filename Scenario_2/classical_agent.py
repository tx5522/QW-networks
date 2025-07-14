"""
Copyright 2018 Alexey Melnikov and Katja Ried.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Alexey Melnikov and Katja Ried, implementing ideas from

'Projective simulation for artificial intelligence'
Hans J. Briegel & Gemma De las Cuevas
Scientific Reports 2, Article number: 400 (2012) doi:10.1038/srep00400

and

'Projective Simulation for Classical Learning Agents: A Comprehensive Investigation'
Julian Mautner, Adi Makmal, Daniel Manzano, Markus Tiersch & Hans J. Briegel
New Generation Computing, Volume 33, Issue 1, pp 69-114 (2015) doi:10.1007/s00354-015-0102-0
"""

# Refactored code by Jiameng Tian.

import numpy as np
from typing import List


class BasicPSAgent:
    """
    Projective Simulation agent with two-layered network.
    Features: forgetting, glow, and optional softmax-based action selection.
    """

    def __init__(
        self,
        num_actions: int,
        num_percepts_list: List[int],
        gamma_damping: float,
        eta_glow_damping: float,
        policy_type: str = "standard",  # "standard" or "softmax"
        beta_softmax: float = 1.0,
    ):
        """
        Initialize the agent.

        Args:
            num_actions: Number of possible actions (>=1).
            num_percepts_list: Cardinality of each perceptual feature (list of ints).
            gamma_damping: Forgetting rate for h-values (0 <= gamma <= 1).
            eta_glow_damping: Damping rate for glow (1 disables glow).
            policy_type: 'standard' or 'softmax' for action selection strategy.
            beta_softmax: Softmax temperature parameter (only used if policy_type='softmax').
        """
        self.num_actions = num_actions
        self.num_percepts_list = num_percepts_list
        self.gamma_damping = gamma_damping
        self.eta_glow_damping = eta_glow_damping
        self.policy_type = policy_type
        self.beta_softmax = beta_softmax

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
        if self.policy_type == "softmax":
            h_scaled = self.beta_softmax * h_vector
            h_shifted = h_scaled - np.max(h_scaled)
            probabilities = np.exp(h_shifted) / np.sum(np.exp(h_shifted))
        else:
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
