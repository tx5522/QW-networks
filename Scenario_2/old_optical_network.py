import torch
import torch.nn as nn


# Half-Wave Plate (HWP) acting on the i-th and j-th modes (light amplitudes are restricted to the real domain)
class HWPBlock(nn.Module):
    def __init__(self, i, j):
        super(HWPBlock, self).__init__()
        self.i = i  # Index of the first mode
        self.j = j  # Index of the second mode
        # self.theta = nn.Parameter(torch.tensor(torch.pi / 8))  # Trainable angle parameter (real number)
        self.theta = nn.Parameter(torch.randn(()) * torch.pi)  # Random initial angle for waveplate selection

    def forward(self, x):
        """
        x: [batch_size, dim] optical mode amplitudes
        """
        # Extract the i-th and j-th components
        xi = x[:, self.i]
        xj = x[:, self.j]
        vec = torch.stack([xi, xj], dim=1)  # Shape: [batch, 2]

        # Construct waveplate transformation matrix (2θ follows optical conventions)
        c = torch.cos(2 * self.theta)
        s = torch.sin(2 * self.theta)

        # Jones matrix
        U = torch.stack([torch.stack([c, s]), torch.stack([s, -c])])  # Shape: [2, 2]

        # Apply transformation
        out_vec = vec @ U.T  # Shape: [batch, 2]

        # Update the original vector
        x = x.clone()
        x[:, self.i] = out_vec[:, 0]
        x[:, self.j] = out_vec[:, 1]

        return x


class Clements(nn.Module):
    def __init__(self, dim, num_layers):
        super(Clements, self).__init__()

        assert dim % 2 == 0, "dim must be even: each mode has [H, V] components."

        self.dim = dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # Construct staggered layers
        for depth in range(num_layers):
            hwp_layer = []
            offset = depth % 2  # Even layers start from index 0, odd layers start from index 1
            for i in range(offset, dim - 1, 2):
                hwp_layer.append(HWPBlock(i, i + 1))
            self.layers.append(nn.ModuleList(hwp_layer))

    def forward(self, x):
        for hwp_layer in self.layers:
            for hwp in hwp_layer:
                x = hwp(x)
        return x

    def get_hwp_angles(self):
        """
        Output HWP angle parameters for each layer, ensuring angles are within [0, π]
        """
        angles = []
        for depth, hwp_layer in enumerate(self.layers):
            layer_angles = []
            for hwp in hwp_layer:
                # Normalize theta to ensure it lies within [0, π]
                normalized_theta = hwp.theta % torch.pi
                layer_angles.append(normalized_theta.item())
            angles.append(layer_angles)
        return angles