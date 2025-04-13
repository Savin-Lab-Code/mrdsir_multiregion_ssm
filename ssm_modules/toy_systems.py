import math
import torch
import torch.nn as nn

from dev.linalg_utils import bmv, bip, bop


class VdpDynamicsModel(nn.Module):
    def __init__(self, bin_sz=5e-3, mu=1.5, tau_1=0.1, tau_2=0.1, scale=1.0):
        super().__init__()

        self.mu = mu
        self.scale = scale
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.bin_sz = bin_sz

    def forward(self, z_t):
        z_t_s = self.scale * z_t
        tau_1_eff = self.bin_sz / self.tau_1
        tau_2_eff = self.bin_sz / self.tau_2

        z_tp1_d0 = z_t[..., 0] + tau_1_eff * z_t[..., 1]
        z_tp1_d1 = z_t[..., 1] + tau_2_eff * (self.mu * (1 - (z_t_s[..., 0])**2) * z_t_s[..., 1] - z_t_s[..., 0]) / self.scale
        z_tp1 = torch.concat([z_tp1_d0[..., None], z_tp1_d1[..., None]], dim=-1)

        return z_tp1


class RingAttractorDynamics(nn.Module):
    def __init__(self, bin_sz=5e-3, d=1.0, w=1.7, device='cpu'):
        super().__init__()

        self.d = d
        self.w = w
        self.bin_sz = bin_sz

    def forward(self, z_t):
        r_t = z_t.pow(2).sum(dim=-1).sqrt()[..., None]
        theta_t = torch.arctan2(z_t[..., 1], z_t[..., 0])[..., None]

        r_tp1 = r_t + self.bin_sz * (self.d - r_t)
        # r_tp1 = r_t + self.bin_sz * r_t * (self.d - r_t**2)
        theta_tp1 = theta_t + self.w * self.bin_sz
        z_tp1 = torch.concat([r_tp1 * torch.cos(theta_tp1), r_tp1 * torch.sin(theta_tp1)], dim=-1)
        return z_tp1


class SpiralDynamics(nn.Module):
    def __init__(self, w=math.pi/16, rho=0.98, device='cpu'):
        super().__init__()
        angle = torch.tensor(w)
        self.A = rho * torch.tensor([[torch.cos(angle), torch.sin(angle)],
                          [-torch.sin(angle), torch.cos(angle)]], device=device)

    def forward(self, z_t):
        return bmv(self.A, z_t)


class DuffingOscillator(nn.Module):
    def __init__(self, delta, beta, alpha, tau_1=0.05, tau_2=0.05, bin_sz=5e-3, device='cpu'):
        super().__init__()
        self.beta = torch.tensor(beta, device=device)
        self.delta = torch.tensor(delta, device=device)
        self.alpha = torch.tensor(alpha, device=device)

        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.bin_sz = bin_sz

    def forward(self, z_t):
        tau_1_eff = self.bin_sz / self.tau_1
        tau_2_eff = self.bin_sz / self.tau_2

        z_tp1_d0 = z_t[..., 0] + tau_1_eff * z_t[..., 1]
        z_tp1_d1 = z_t[..., 1] - tau_2_eff * (self.delta * z_t[..., 1] + z_t[..., 0] * (self.beta + self.alpha * z_t[..., 0]**2))
        z_tp1 = torch.concat([z_tp1_d0[..., None], z_tp1_d1[..., None]], dim=-1)
        return z_tp1
