import torch
import torch.nn as nn


class ThreeRegionInputModule(nn.Module):
    def __init__(self, n_neurons_K, dyn_readout_fn_KK):
        super(ThreeRegionInputModule, self).__init__()

        self.K = len(dyn_readout_fn_KK)
        self.n_neurons_K = n_neurons_K
        self.dyn_readout_fn_KK = nn.ModuleList(dyn_readout_fn_KK)

    def get_likelihood_term(self, u_K0, u_KK):
        u_proj = []

        for k in range(self.K):
            u_proj.append(torch.zeros(self.n_neurons_K[k], device=u_KK[0].device))

        return u_proj

    def get_dynamics_term(self, u_K0, u_KK):
        u_proj = []

        for k in range(self.K):
            u_proj.append(self.dyn_readout_fn_KK[k](u_KK[k]))

        return u_proj