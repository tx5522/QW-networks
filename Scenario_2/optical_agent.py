import os

# Allow numpy and torch to import different versions of the MKL library
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import math
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Half-Wave Plate (HWP) layer
class HWPLayer(nn.Module):
    def __init__(self, dim: int, offset: int):
        """
        dim: Dimension of the input vector
        offset: Offset value determining whether the coupling pairs in this layer start at 0 or 1
        """
        super(HWPLayer, self).__init__()
        self.dim = dim
        self.pairs = [(i, i + 1) for i in range(offset, dim - 1, 2)]
        self.num_pairs = len(self.pairs)
        self.theta = nn.Parameter(torch.randn(self.num_pairs) * torch.pi)
        # self.theta = nn.Parameter(torch.tensor(torch.pi / 8))  # Trainable angle parameter (real number)

    def forward(self, x: torch.Tensor):
        """
        x: [batch_size, dim] Optical amplitudes
        """
        y = x.clone()
        c = torch.cos(2 * self.theta)  # [num_pairs]
        s = torch.sin(2 * self.theta)  # [num_pairs]

        for idx, (i, j) in enumerate(self.pairs):
            xi = x[:, i]
            xj = x[:, j]
            # Apply 2x2 orthogonal transformation
            y[:, i] = c[idx] * xi + s[idx] * xj
            y[:, j] = s[idx] * xi - c[idx] * xj

        return y


# Construct arbitrary orthogonal matrices using HWP layers
class Clements(nn.Module):
    def __init__(self, dim, num_layers):
        super(Clements, self).__init__()

        assert dim % 2 == 0, "dim must be even: each mode has [H, V] components."

        self.dim = dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # Build staggered HWPLayers
        for depth in range(num_layers):
            offset = depth % 2  # Even layers start at 0, odd layers start at 1
            self.layers.append(HWPLayer(dim, offset))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_hwp_angles(self):
        """
        Output HWP angle parameters for each layer, ensuring angles are within [0, π]
        """
        angles = []
        for layer in self.layers:
            normalized_theta = layer.theta % torch.pi  # Ensure in [0, π]
            angles.append(normalized_theta.tolist())
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
            num_actions (int): Number of discrete actions available.
            num_percepts_list (List[int]): Number of discrete values for each perceptual dimension, 
                e.g., [2, 3] indicates two perceptual dimensions with 2 and 3 possible values respectively.
            eta_glow_damping (float): Glow damping factor in (0, 1), controlling memory forgetting speed.
            dim (int): Input dimension of the optical network.
            num_layers (int): Number of layers in the optical network.
            encoding_scheme (str): Input encoding method, either "one_hot" or "superposition"
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
        ), f"dim must be ≥ required dimension {expected_dim} (determined by encoding scheme)"

        # Initialize glow matrix
        self.g_matrix = torch.zeros(
            (num_actions, self.num_percepts), dtype=torch.float32, requires_grad=False
        )

        # Pre-generate perceptual vectors
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

        # Cache state-action probabilities
        self.action_prob_cache = {}

    def _build_superposition_percept_inputs(self) -> torch.Tensor:
        """
        Construct superposition perceptual vectors, supporting arbitrary-dimensional discrete perceptual spaces.
        Each vector has exactly len(num_percepts_list) non-zero entries, each 1/sqrt(k),
        located at the value indices of each dimension (considering offset superposition).
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
        Encode multi-dimensional discrete percepts into a unique state index.
        """
        index = 0
        multiplier = 1
        for val, cardinality in zip(observation, self.num_percepts_list):
            index += val * multiplier
            multiplier *= cardinality
        return index

    def _decode_percept_index(self, index: int) -> List[int]:
        """
        Decode a unique state index back into a multi-dimensional discrete percept vector.
        """
        observation = []
        for cardinality in self.num_percepts_list:
            observation.append(index % cardinality)
            index //= cardinality
        return observation

    def _update_glow_matrix(self, action: int, percept: int):
        """
        Update the glow matrix: dampen all entries while activating the current state-action pair.
        """
        with torch.no_grad():
            self.g_matrix *= 1.0 - self.eta_glow_damping
            self.g_matrix[action, percept] = 1.0

    def _compute_action_probabilities(self, percept: int) -> torch.Tensor:
        """
        Compute the action selection probability distribution for a given perceptual state.
        """
        input_vec = self.percept_inputs[[percept]]
        output = self.clements(input_vec)[:, : self.num_actions].reshape(-1)
        intensity = output**2 + self.eps  # Prevent division by zero
        return intensity / intensity.sum()

    def select_action(self, observation: List[int], epsilon: float) -> int:
        """
        Select action using ε-greedy strategy and update glow.

        Args:
            observation (List[int]): Current percept (list of discrete values)
            epsilon (float): Exploration probability

        Returns:
            int: Index of the selected action
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

        self._update_glow_matrix(action, percept)
        return action

    def learn_from_reward(self, reward: float, max_iters: int = 50, tol: float = 1e-4):
        """
        Train using the glow matrix upon receiving positive rewards, minimizing KL divergence.

        Args:
            reward (float): Current reward (only positive values trigger learning)
            max_iters (int): Maximum number of iterations
            tol (float): Convergence threshold for KL divergence
        """
        if reward <= 0:
            return

        mask = self.g_matrix.sum(0) != 0  # Which states participated in glow
        if not mask.any():
            return

        input_batch = self.percept_inputs[mask]
        g_batch = self.g_matrix[:, mask].T  # [B, num_actions]

        # Reward-guided target distribution
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

            print(f"Iter {iter_idx}: loss = {loss.item():.6f}")
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


if __name__ == "__main__":

    # Generate a random orthogonal matrix Q
    def random_orthogonal_matrix(dim):
        # Generate orthogonal matrix using QR decomposition
        A = torch.randn(dim, dim)
        Q, _ = torch.linalg.qr(A)
        return Q

    # Create training data
    def generate_data(Q, num_samples=1000):
        X = torch.randn(num_samples, Q.size(0))
        Y = X @ Q
        return X, Y

    # Define number of optical modes and layers
    dim = 6
    num_layers = 6

    # Initialize network
    net = Clements(dim, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-2)

    # Build target orthogonal matrix and training data
    Q_target = random_orthogonal_matrix(dim)
    X_train, Y_train = generate_data(Q_target, num_samples=1000)

    # Training loop
    for epoch in range(1000):
        optimizer.zero_grad()
        output = net(X_train)
        loss = criterion(output, Y_train)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    # Testing
    X_test, Y_test = generate_data(Q_target, num_samples=100)
    Y_pred = net(X_test)

    # Evaluate error
    test_loss = F.mse_loss(Y_pred, Y_test).item()
    print(f"\nFinal Test MSE: {test_loss:.6e}")

    print(net.get_hwp_angles())