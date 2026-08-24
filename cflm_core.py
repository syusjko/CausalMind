import torch
import torch.nn as nn

class DualMemoryCFLM(nn.Module):
    def __init__(self, state_dim=12, control_dim=4):
        super().__init__()
        # Slow Weights (theta_fixed) - Represents frozen core knowledge
        self.theta_fixed = nn.Linear(state_dim, control_dim)
        
        # Fast Weights (theta_plastic) - Dynamic cache for new causal relationships
        self.theta_plastic = nn.Linear(state_dim, control_dim)
        
        self.tau = 0.5  # Surprisal threshold
        self.alpha = 1.0 # Endocrine gating scale
        
    def forward(self, z0, surprisal):
        """
        Continuous Fractal Language Model (CFLM) approximation.
        ODE integration: z_out = z_0 + int_0^1 (f_fractal + alpha * ReLU(S_t - tau) * g) dt
        """
        # Fractal flow from core knowledge
        f_fractal = self.theta_fixed(z0)
        
        # Endocrine gating mechanism activated by Cognitive Surprisal
        gate = self.alpha * torch.relu(surprisal - self.tau)
        g_plastic = self.theta_plastic(z0)
        
        # Assuming dt=1 for discrete approximation (15M param scale real-time O(1))
        # z_out handles 4D control vector: Pitch, Roll, Yaw, Throttle
        z_out = z0[:, :4] + f_fractal + gate * g_plastic 
        return z_out
