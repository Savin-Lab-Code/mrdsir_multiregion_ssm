import copy
import math
import torch
import einops
import optree
import numpy as np
import torch.nn as nn
import dev.utils as utils
import torch.nn.functional as Fn

from itertools import chain
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


def c_tilde_fn(k, c):
    # k = kc[0]
    # c = kc[1]
    return c if c < k else c+1


def pack_dense_multiregion_state(z_K, H_KK):
    n_regions = len(z_K)
    z_K_vec = torch.cat(z_K, dim=-1)
    H_KK_rows_cat = [torch.cat(H_KK[k], dim=-1) for k in range(n_regions)]
    H_KK_vec = torch.cat(H_KK_rows_cat, dim=-1)
    z = torch.cat([z_K_vec, H_KK_vec], dim=-1)
    return z


def unpack_dense_multiregion_state(z, L_K, iir_order):
    K = len(L_K)
    L_K_sum = sum(L_K)

    z_K = [z[..., sum(L_K[:k]): sum(L_K[:k]) + L_K[k]] for k in range(K)]
    H_KK = []

    start_dx = L_K_sum
    for k in range(K):
        H_KK.append([])

        for c, c_tilde in zip(range(K - 1), chain(range(k), range(k + 1, K))):
            H_KK[k].append(z[..., start_dx: start_dx + L_K[c_tilde] * iir_order * 2])
            start_dx += L_K[c_tilde] * iir_order * 2

    return z_K, H_KK


def unpack_z_K(z_K_vec, L_K):
    z_K = [z_K_vec[..., sum(L_K[:k]): sum(L_K[:k]) + L_K[k]] for k in range(len(L_K))]
    return z_K


def stabilizeFcComplexDiagonalDynamics(dynamics_mod, device='cpu', ub=0.9, angle_ub=math.pi, eps=1e-2):
    with torch.no_grad():
        for k, row_k in enumerate(dynamics_mod.mean_fn.transition_H_KK):
            for l, H_kl in enumerate(row_k):
                roots = H_kl.roots.data.clone()

                r_mask_big = H_kl.roots[:, 0] >= 1
                r_mask_sml = H_kl.roots[:, 0] < ub
                roots[r_mask_big, 0] = 1.0 - eps
                roots[r_mask_sml, 0] = ub

                theta_mask_big = H_kl.roots[:, 1] >= angle_ub
                theta_mask_sml = H_kl.roots[:, 1] < 0.0
                roots[theta_mask_big, 1] = angle_ub - eps
                roots[theta_mask_sml, 1] = 0.0
                H_kl.roots.data = roots


def stabilize_directly_connected_dynamics(dynamics_mod):
    with torch.no_grad():
        for k in range(len(dynamics_mod.mean_fn.transition_z_K)):
            if isinstance(dynamics_mod.mean_fn.transition_z_K[k], nn.Linear):
                F = dynamics_mod.mean_fn.transition_z_K[k].weight.data
                U, S, VmT = torch.linalg.svd(F, full_matrices=True)
                dynamics_mod.mean_fn.transition_z_K[k].weight.data = (U * S.clip(max=1.0)) @ VmT


class ComplexDiagonalDynamics(nn.Module):
    def __init__(self, n_latents, iir_order, device='cpu'):
        super(ComplexDiagonalDynamics, self).__init__()
        # roots are represented as (radius, angle)

        self.iir_order = iir_order
        self.n_latents = n_latents
        self.roots = torch.nn.Parameter(torch.rand((iir_order, 2), device=device))

        # initialize at low frequencies [0, pi/3], and slowish decay radius [0.9, 1]
        with torch.no_grad():
            self.roots.data[:, 0] = 0.9 + 0.1 * torch.rand_like(self.roots.data[:, 0])
            self.roots.data[:, 1] = (math.pi / 6) + (math.pi / 6) * torch.rand_like(self.roots.data[:, 1])

    def forward(self, z_t):
        roots = torch.zeros_like(self.roots)
        roots[:, 0] += self.roots[:, 0] * torch.cos(self.roots[:, 1])
        roots[:, 1] += self.roots[:, 0] * torch.sin(self.roots[:, 1])
        roots_repeat = einops.repeat(roots, 'n d -> (n c) d', c=self.n_latents)
        re_z_t = z_t[..., 0::2]
        im_z_t = z_t[..., 1::2]

        z_tp1 = torch.zeros_like(z_t)
        z_tp1[..., 0::2] += re_z_t * roots_repeat[:, 0] - im_z_t * roots_repeat[:, 1]
        z_tp1[..., 1::2] += re_z_t * roots_repeat[:, 1] + im_z_t * roots_repeat[:, 0]

        return z_tp1


class FullyConnectedDynamics(nn.Module):
    def __init__(self, transition_z_K, transition_H_KK, readout_fn_z_KK, readout_fn_H_KK, n_regions, n_latents_K,
                 iir_order):
        super(FullyConnectedDynamics, self).__init__()

        self.iir_order = iir_order
        self.n_regions = n_regions
        self.n_latents_K = n_latents_K
        self.n_latents_K_sum = sum(n_latents_K)

        self.transition_H_KK = nn.ModuleList(transition_H_KK)
        self.readout_fn_z_KK = nn.ModuleList(readout_fn_z_KK)
        self.readout_fn_H_KK = nn.ModuleList(readout_fn_H_KK)

        self.transition_z_K = nn.ModuleList(transition_z_K)

        # self.transition_z_K_ = list(self.transition_z_K)
        # self.readout_fn_H_KK_ = [list(readout_fn_H_KK_k) for readout_fn_H_KK_k in self.readout_fn_H_KK]
        # self.transition_H_KK_ = [list(transition_H_KK_k) for transition_H_KK_k in self.transition_H_KK]
        # self.transition_H_KK_ = [[transition_H_KK_lk for transition_H_KK_lk in transition_H_KK_l] for transition_H_KK_l in self.transition_H_KK]

    def set_linear_dynamics(self):
        self.transition_z_K = self.transition_z_K_ln

    def set_nonlinear_dynamics(self):
        self.transition_z_K = self.transition_z_K_nl

    def forward_packed(self, z):
        z_K, H_KK = unpack_dense_multiregion_state(z, self.n_latents_K, self.iir_order)
        z_K_tp1, H_KK_tp1 = self.forward_unpacked(z_K, H_KK)
        z_tp1 = pack_dense_multiregion_state(z_K_tp1, H_KK_tp1)
        return z_tp1

    def forward_unpacked(self, z_K, H_KK):
        z_K_tp1 = [self.transition_z_K[k](z_K[k]) for k in range(self.n_regions)]
        # z_K_tp1 = optree.tree_map(lambda x, y: x(y), self.transition_z_K_, z_K)

        H_KK_tp1 = []
        for k in range(self.n_regions):
            H_KK_tp1.append([])
            for l in range(self.n_regions-1):
                H_KK_tp1[k].append(self.transition_H_KK[k][l](H_KK[k][l]))
        # H_KK_tp1 = optree.tree_map(lambda x, y: x(y), self.transition_H_KK_, H_KK)

        for k in range(self.n_regions):
            for c, c_tilde in zip(range(self.n_regions - 1), chain(range(k), range(k + 1, self.n_regions))):
                H_KK_tp1[k][c][..., ::2] += self.readout_fn_z_KK[k][c](z_K[c_tilde])
        # readout_z_fn = lambda kc: self.readout_fn_z_KK[kc[0]][kc[1]](z_K[c_tilde_fn(kc[0], kc[1])])
        # kc_array = [[np.array([i, j], dtype=int) for j in range(self.n_regions - 1)] for i in range(self.n_regions)]
        # H_KK_tp1_delta_real = tree_map(readout_z_fn, kc_array)
        # H_KK_tp1_real = optree.tree_map(lambda x, y: x[..., ::2] + y, H_KK_tp1, H_KK_tp1_delta_real)
        # H_KK_tp1_reim = optree.tree_map(lambda x, y: torch.stack([y, x[..., 1::2]], dim=-1).flatten(start_dim=-2, end_dim=-1),
        #                                 H_KK_tp1, H_KK_tp1_real)

        for k in range(self.n_regions):
            for l in range(self.n_regions - 1):
                if self.readout_fn_H_KK[k][l].weight.data.sum() == 0:
                    continue
                z_add = self.readout_fn_H_KK[k][l](H_KK_tp1[k][l][..., ::2])
                z_K_tp1[k] += z_add
        # z_K_delta_tp1 = optree.tree_map(lambda x, y: x(y), self.readout_fn_H_KK_, H_KK_tp1_real)
        # for k in range(self.n_regions):
        #     z_K_tp1[k] += sum(z_K_delta_tp1[k])

        return z_K_tp1, H_KK_tp1


class FullyConnectedMultiRegionDenseGaussianInitialCondition(nn.Module):
    def __init__(self, iir_order, m_K_init, Q_K_init_diag, Q_H_KK_init_diag, device='cpu'):
        super(FullyConnectedMultiRegionDenseGaussianInitialCondition, self).__init__()

        self.device = device
        self.iir_order = iir_order
        self.K = len(Q_K_init_diag)
        self.L_K = [len(Q_k_diag) for Q_k_diag in Q_K_init_diag]

        self.m_K_init = torch.nn.ParameterList(m_K_init).to(self.device)
        self.log_Q_K = torch.nn.ParameterList([utils.softplus_inv(Q_K_init_diag[k]) for k in range(self.K)]).to(device)
        self.log_Q_H_KK = torch.nn.ParameterList([torch.nn.ParameterList([utils.softplus_inv(Q_H_KK_init_diag[l][k])
                                                                          for k in range(self.K-1)]).to(device)
                                                  for l in range(self.K)])

    def get_Q_init(self):
        Q_K_diag_vec = Fn.softplus(torch.cat(list(self.log_Q_K), dim=-1))
        Q_H_KK_rows_vec = [Fn.softplus(torch.cat(list(self.log_Q_H_KK[k]), dim=-1)) for k in range(self.K)]
        Q_H_KK_diag_vec = torch.cat(list(Q_H_KK_rows_vec), dim=-1)
        Q_diag = torch.cat([Q_K_diag_vec, Q_H_KK_diag_vec], dim=-1)
        return Q_diag

    def get_m_init(self, device):
        m_K_init = torch.cat(list(self.m_K_init), dim=-1)
        m_H_KK_vec = torch.zeros(2*self.iir_order*sum(self.L_K)*(self.K - 1), device=device)
        return torch.cat([m_K_init, m_H_KK_vec], dim=-1)


class FullyConnectedDenseGaussianDynamics(nn.Module):
    def __init__(self, iir_order, mean_fn, Q_K_diag, device='cpu'):
        super(FullyConnectedDenseGaussianDynamics, self).__init__()

        self.device = device
        self.K = len(Q_K_diag)
        self.iir_order = iir_order
        self.L_K = [len(Q_k_diag) for Q_k_diag in Q_K_diag]

        self.mean_fn = mean_fn
        self.log_Q_K = torch.nn.ParameterList([utils.softplus_inv(Q_K_diag[k]) for k in range(self.K)]).to(device)

    def get_Q(self, device):
        Q_K_diag_vec = Fn.softplus(torch.cat(list(self.log_Q_K), dim=-1))
        H_KK_diag_vec = torch.zeros(2*self.iir_order*sum(self.L_K)*(self.K-1), device=device)
        Q_diag = torch.cat([Q_K_diag_vec, H_KK_diag_vec], dim=-1)
        return Q_diag


class DirectlyConnectedDynamics(nn.Module):
    def __init__(self, transition_z_K, transition_z_KK, n_regions, n_latents_K):
        super(DirectlyConnectedDynamics, self).__init__()

        self.n_regions = n_regions
        self.n_latents_K = n_latents_K
        self.n_latents_K_sum = sum(n_latents_K)
        self.transition_z_K = nn.ModuleList(transition_z_K)
        self.transition_z_KK = nn.ModuleList(transition_z_KK)

    def forward_packed(self, z):
        z_K = unpack_z_K(z, self.n_latents_K)
        z_K_tp1 = self.forward_unpacked(z_K)
        z_tp1 = torch.cat(z_K_tp1, dim=-1)
        return z_tp1

    def forward_unpacked(self, z_K):
        z_K_tp1 = [self.transition_z_K[k](z_K[k]) for k in range(self.n_regions)]

        for k in range(self.n_regions):
            for c, c_tilde in zip(range(self.n_regions - 1), chain(range(k), range(k + 1, self.n_regions))):
                z_K_tp1[k] += self.transition_z_KK[k][c](z_K[c_tilde])
        return z_K_tp1


class DirectlyConnectedDenseGaussianDynamics(nn.Module):
    def __init__(self, mean_fn, Q_K_diag, device='cpu'):
        super(DirectlyConnectedDenseGaussianDynamics, self).__init__()

        self.device = device
        self.K = len(Q_K_diag)
        self.L_K = [len(Q_k_diag) for Q_k_diag in Q_K_diag]

        self.mean_fn = mean_fn
        self.log_Q_K = torch.nn.ParameterList([utils.softplus_inv(Q_K_diag[k]) for k in range(self.K)]).to(device)

    def get_Q(self, device):
        Q_K_diag_vec = Fn.softplus(torch.cat(list(self.log_Q_K), dim=-1))
        return Q_K_diag_vec


class DirectlyConnectedMultiRegionDenseGaussianInitialCondition(nn.Module):
    def __init__(self, m_K_init, Q_K_init_diag, device='cpu'):
        super(DirectlyConnectedMultiRegionDenseGaussianInitialCondition, self).__init__()

        self.device = device
        self.K = len(Q_K_init_diag)
        self.L_K = [len(Q_k_diag) for Q_k_diag in Q_K_init_diag]

        self.m_K_init = torch.nn.ParameterList(m_K_init).to(self.device)
        self.log_Q_K = torch.nn.ParameterList([utils.softplus_inv(Q_K_init_diag[k])
                                               for k in range(self.K)]).to(device)

    def get_Q_init(self):
        Q_K_diag_vec = Fn.softplus(torch.cat(list(self.log_Q_K), dim=-1))
        return Q_K_diag_vec

    def get_m_init(self, device):
        m_K_init = torch.cat(list(self.m_K_init), dim=-1)
        return m_K_init
