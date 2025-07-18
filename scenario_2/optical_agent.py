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

import os

# Allow numpy and torch to load different versions of the MKL library
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Half-Wave Plate (HWP) layer; each logical HWP layer consists of two physical HWP layers
class HWPLayer(nn.Module):
    def __init__(self, dim: int):
        """
        dim: Input vector dimension
        """
        super(HWPLayer, self).__init__()

        assert dim % 2 == 0, "dim must be even: each mode has [H, V] components."

        self.dim = dim

        # Input is [A1 B1 A2 B2 ... An Bn]
        # First HWPs apply to (A1 B1), (A2 B2), ..., (An Bn)
        # Output is [C1 D1 C2 D2 ... Cn Dn]
        self.pairs_front = [(i, i + 1) for i in range(0, dim - 1, 2)]
        self.num_pairs_front = len(self.pairs_front)
        self.theta_front = nn.Parameter(torch.randn(self.num_pairs_front) * torch.pi)

        # First BD (beam displacer) operation: reorder the vector, swap in pairs except ends
        # [C1 D1 C2 D2 C3 D3 ... Cn Dn] -> [C1 C2 D1 C3 D2 ... Cn Dn-1 Dn]
        indices = torch.arange(self.dim)
        self.swap_indices1 = indices.clone()
        for i in range(1, dim - 1, 2):
            self.swap_indices1[i] = indices[i + 1]
            self.swap_indices1[i + 1] = indices[i]

        # Input is [C1 C2 D1 C3 D2 ... Cn Dn-1 Dn]
        # Second HWPs apply to (C2 D1), (C3 D2), ..., (Cn Dn-1)
        self.pairs_rear = [(i, i + 1) for i in range(1, dim - 1, 2)]
        self.num_pairs_rear = len(self.pairs_rear)
        self.theta_rear = nn.Parameter(torch.randn(self.num_pairs_rear) * torch.pi)

        # Second BD operation: swap in pairs starting from the beginning
        self.swap_indices2 = indices.clone()
        for i in range(0, dim - 1, 2):
            self.swap_indices2[i] = indices[i + 1]
            self.swap_indices2[i + 1] = indices[i]

    def forward(self, x: torch.Tensor):
        """
        x: [batch_size, dim] Optical amplitude
        """

        y = x.clone()
        c = torch.cos(2 * self.theta_front)
        s = torch.sin(2 * self.theta_front)
        for idx, (i, j) in enumerate(self.pairs_front):
            xi = x[:, i]
            xj = x[:, j]
            y[:, i] = c[idx] * xi + s[idx] * xj
            y[:, j] = s[idx] * xi - c[idx] * xj

        y = y[:, self.swap_indices1]

        z = y.clone()
        c = torch.cos(2 * self.theta_rear)
        s = torch.sin(2 * self.theta_rear)
        for idx, (i, j) in enumerate(self.pairs_rear):
            yi = y[:, i]
            yj = y[:, j]
            z[:, i] = c[idx] * yi + s[idx] * yj
            z[:, j] = s[idx] * yi - c[idx] * yj

        z = z[:, self.swap_indices2]
        return z


# Use HWPs to construct an arbitrary orthogonal matrix (Clements decomposition)
class Clements(nn.Module):
    def __init__(self, dim, num_layers):
        """
        Construct a Clements network with num_layers alternating HWPLayers
        Input is [H1, V1, H2, V2, ..., Hn Vn], so dim must be even
        """
        super(Clements, self).__init__()

        assert dim % 2 == 0, "dim must be even: each mode has [H, V] components."

        self.dim = dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([HWPLayer(dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_hwp_angles(self):
        """
        Return the HWP angles of each layer, normalized to [0, π]
        """
        angles = []
        for layer in self.layers:
            normalized_theta_front = layer.theta_front % torch.pi  # ensure in [0, π]
            normalized_theta_rear = layer.theta_rear % torch.pi
            angles.append(normalized_theta_front.tolist())
            angles.append(normalized_theta_rear.tolist())
        return angles


class OpticalAgent:
    def __init__(
        self,
        num_actions: int,
        num_percepts_list: List[int],
        eta_glow_damping: float,
        dim: int,
        num_layers: int,
        encoding_scheme: str = "one_hot",
    ):
        """
        Initialize the optical agent.

        Args:
            num_actions (int): Number of discrete actions.
            num_percepts_list (List[int]): Number of discrete values per percept dimension (e.g., [2, 3]).
            eta_glow_damping (float): Glow decay factor in (0, 1), controls memory decay speed.
            dim (int): Input dimension of the optical network.
            num_layers (int): Number of optical layers.
            encoding_scheme (str): Encoding method: "one_hot" or "superposition"
        """
        self.num_actions = num_actions
        self.num_percepts_list = num_percepts_list
        self.num_percepts = math.prod(num_percepts_list)
        self.eta_glow_damping = eta_glow_damping
        self.dim = dim
        self.num_layers = num_layers
        self.encoding_scheme = encoding_scheme
        self.eps = 1e-6

        expected_dim = (
            self.num_percepts
            if encoding_scheme == "one_hot"
            else sum(num_percepts_list)
        )
        assert (
            dim >= expected_dim
        ), f"dim must be ≥ expected dimension {expected_dim} (based on encoding scheme)"

        # Initialize glow matrix
        self.g_matrix = torch.zeros(
            (num_actions, self.num_percepts), dtype=torch.float32, requires_grad=False
        )

        # Pre-generate percept vectors
        self.percept_inputs = (
            torch.eye(
                self.num_percepts, self.dim, dtype=torch.float32, requires_grad=False
            )
            if encoding_scheme == "one_hot"
            else self._build_superposition_percept_inputs()
        )

        # Initialize optical network
        self.clements = Clements(self.dim, self.num_layers)
        self.optimizer = optim.Adam(self.clements.parameters(), lr=0.01)

        # Cache for percept-action probabilities
        self.action_prob_cache = {}

    def _build_superposition_percept_inputs(self) -> torch.Tensor:
        """
        Build superposition percept vectors for arbitrary discrete percept spaces.
        Each vector has len(num_percepts_list) non-zero entries (1/sqrt(k)),
        placed at the respective indices of each percept dimension.
        """
        sqrt_k = 1.0 / math.sqrt(len(self.num_percepts_list))

        inputs = torch.zeros((self.num_percepts, self.dim), dtype=torch.float32)

        for state_index in range(self.num_percepts):
            obs = self._decode_percept_index(state_index)
            offset = 0
            for i, val in enumerate(obs):
                inputs[state_index, offset + val] = sqrt_k
                offset += self.num_percepts_list[i]

        return inputs

    def _encode_percept(self, observation: List[int]) -> int:
        """
        Encode a multi-dimensional discrete percept into a unique state index.
        """
        index = 0
        multiplier = 1
        for val, cardinality in zip(observation, self.num_percepts_list):
            index += val * multiplier
            multiplier *= cardinality
        return index

    def _decode_percept_index(self, index: int) -> List[int]:
        """
        Decode a unique state index back into a multi-dimensional percept vector.
        """
        observation = []
        for cardinality in self.num_percepts_list:
            observation.append(index % cardinality)
            index //= cardinality
        return observation

    def _update_glow_matrix(self, action: int, percept: int):
        """
        Update the glow matrix: decay all entries, and activate the current (state, action) pair.
        """
        with torch.no_grad():
            self.g_matrix *= 1.0 - self.eta_glow_damping
            self.g_matrix[action, percept] = 1.0

    def _compute_action_probabilities(self, percept: int) -> torch.Tensor:
        """
        Compute the action selection probabilities for a given percept.
        """
        input_vec = self.percept_inputs[[percept]]
        output = self.clements(input_vec)[:, : self.num_actions].reshape(-1)
        intensity = output**2 + self.eps  # avoid divide-by-zero
        return intensity / intensity.sum()

    def select_action(self, observation: List[int], epsilon: float) -> int:
        """
        ε-greedy action selection strategy, and update glow.

        Args:
            observation (List[int]): Current percept (list of discrete values)
            epsilon (float): Exploration probability

        Returns:
            int: Selected action index
        """
        percept = self._encode_percept(observation)

        if torch.rand(1).item() < epsilon:
            action = torch.randint(0, self.num_actions, (1,)).item()
        else:
            if percept not in self.action_prob_cache:
                with torch.no_grad():
                    self.action_prob_cache[percept] = (
                        self._compute_action_probabilities(percept)
                    )
            probs = self.action_prob_cache[percept]
            action = torch.multinomial(probs, num_samples=1).item()

        action = int(action)
        self._update_glow_matrix(action, percept)
        return action

    def learn_from_reward(self, reward: float, max_iters: int = 50, tol: float = 5e-3):
        """
        After receiving positive reward, train using the glow matrix to minimize KL divergence.

        Args:
            reward (float): Reward signal (training only triggered for positive reward)
            max_iters (int): Maximum number of iterations
            tol (float): KL divergence convergence threshold
        """
        if reward <= 0:
            return

        mask = self.g_matrix.sum(0) != 0  # Which percepts are involved in glow
        if not mask.any():
            return

        input_batch = self.percept_inputs[mask]
        g_batch = self.g_matrix[:, mask].T  # [B, num_actions]

        # Target distribution guided by reward
        output = self.clements(input_batch)[:, : self.num_actions]
        intens = output**2 + self.eps
        p_batch = intens / intens.sum(dim=1, keepdim=True)
        q_batch = p_batch.detach() + reward * g_batch
        q_batch /= q_batch.sum(dim=1, keepdim=True)

        for iter_idx in range(max_iters):
            output = self.clements(input_batch)[:, : self.num_actions]
            intens = output**2 + self.eps
            p_batch = intens / intens.sum(dim=1, keepdim=True)

            kl_div = torch.sum(
                q_batch * (torch.log(q_batch) - torch.log(p_batch)), dim=1
            )
            loss = kl_div.mean()

            # print(f"Iter {iter_idx}: loss = {loss.item():.6f}")
            if loss.item() < tol:
                break

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.clements.parameters(), 1.0)
            self.optimizer.step()

        # Clear cache after update
        self.action_prob_cache.clear()

    def reset_glow(self):
        """
        Clear the glow memory matrix (typically called at the end of an episode).
        """
        self.g_matrix.zero_()
