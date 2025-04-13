import math
import copy
import torch
import scipy
import numpy as np
import torch.nn as nn
import dev.prob_utils as prob_utils

from itertools import chain
from einops import rearrange
from scipy.signal import ss2tf
from sklearn.decomposition import FactorAnalysis
from dev.linalg_utils import bmv, chol_bmv_solve, bqp, bip
from torcheval.metrics.functional import r2_score



def softplus_inv(x):
    #-- from https://github.com/tensorflow/probability/blob/v0.23.0/tensorflow_probability/python/math/generic.py#L531-L582
    if isinstance(x, torch.Tensor):
        return x + torch.log(-torch.expm1(-x))
    else:
        return np.log(np.exp(x) - 1 + 1e-10)


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_stable_nn_linear(L_in, L_out, device='cpu', min_clip=0.0, max_clip=0.99):
    linear = nn.Linear(L_in, L_out, bias=False, device=device)
    A = torch.randn((L_out, L_in), device=device)
    U, S, VmT = torch.linalg.svd(A, full_matrices=False)
    A = U @ torch.diag(S.clip(min=min_clip, max=max_clip)) @ VmT
    linear.weight.data = A
    return linear



def ssm_get_BC(n_latents_in, n_latents_out, iir_order, readout_H_fn, readout_z_fn):
    B = torch.zeros((2*iir_order*n_latents_in, n_latents_in))
    C = torch.zeros((n_latents_out, 2*iir_order*n_latents_in))

    for k in range(n_latents_in*iir_order):
        B[2*k, :] += readout_z_fn.weight[k, :]

    for k in range(n_latents_in*iir_order):
        C[:, 2*k] += readout_H_fn.weight[:, k]

    return B, C


def ssm_set_B_eye(ssm):
    for row_fns in ssm.dynamics_mod.mean_fn.readout_fn_z_KK:
        for readout_fn in row_fns:
            n_in = readout_fn.weight.shape[1]
            n_p = readout_fn.weight.shape[0] // n_in
            readout_fn.weight.requires_grad_(False)

            for n in range(n_p):
                readout_fn.weight.data[n_in*n: n_in*(n+1)] = torch.eye(n_in, dtype=readout_fn.weight.dtype,
                                                                       device=readout_fn.weight.device)
                readout_fn.weight.data[n_in * n: n_in * (n + 1)] /= math.sqrt(n_in)


def readout_fn_z_KK_set_B_eye(readout_fn_z_KK):
    for row_fns in readout_fn_z_KK:
        for readout_fn in row_fns:
            n_in = readout_fn.weight.shape[1]
            n_p = readout_fn.weight.shape[0] // n_in
            readout_fn.weight.requires_grad_(False)

            for n in range(n_p):
                readout_fn.weight.data[n_in*n: n_in*(n+1)] = torch.eye(n_in, dtype=readout_fn.weight.dtype,
                                                                       device=readout_fn.weight.device)
                readout_fn.weight.data[n_in * n: n_in * (n + 1)] /= math.sqrt(n_in)



def ssm_set_C_equ(ssm):
    for row_fns in ssm.dynamics_mod.mean_fn.readout_fn_H_KK:
        for readout_fn in row_fns:
            n_out = readout_fn.weight.shape[0]
            n_p = readout_fn.weight.shape[1] // n_out

            for n in range(n_p):
                int_vals = torch.randint_like(readout_fn.weight.data[:, n_out*n: n_out*(n+1)], -1, 2)
                readout_fn.weight.data[:, n_out * n: n_out * (n + 1)] = int_vals / np.sqrt(readout_fn.weight.shape[1])
                # readout_fn.weight.data[n_in*n: n_in*(n+1)] = torch.eye(n_in, dtype=readout_fn.weight.dtype,
                #                                                        device=readout_fn.weight.device)


def convert_region_dynamics_ssm_ABC(n_latents_in, n_latents_out, transition_fn, readout_H_fn, readout_z_fn, T_iir=250):
    roots = torch.zeros_like(transition_fn.roots)
    roots[:, 0] = transition_fn.roots[:, 0] * torch.cos(transition_fn.roots[:, 1])
    roots[:, 1] = transition_fn.roots[:, 0] * torch.sin(transition_fn.roots[:, 1])

    iir_order, _ = roots.shape

    A_np = np.zeros((iir_order*n_latents_in, n_latents_in*iir_order)) + 0.j
    A = torch.zeros((2*iir_order*n_latents_in, 2*iir_order*n_latents_in))
    B = torch.zeros((2*iir_order*n_latents_in, n_latents_in))
    C = torch.zeros((n_latents_out, 2*iir_order*n_latents_in))
    # D = torch.zeros((n_latents_out, n_latents_out))
    dx_offset = 0
    dx_np = 0

    for k in range(iir_order):
        for l in range(n_latents_in):
            dx_np = k * n_latents_in + l
            dx = 2 * k * n_latents_in + 2 * l
            A[dx, dx] = roots[k, 0]
            A[dx+1, dx+1] = roots[k, 0]
            A[dx, dx+1] = -roots[k, 1]
            A[dx+1, dx] = roots[k, 1]

            A_np[dx_np, dx_np] = roots[k, 0].cpu().detach().numpy() + roots[k, 1].cpu().detach().numpy() * 1j

        dx_offset += 2 * n_latents_in
        dx_np += n_latents_in

    for k in range(n_latents_in*iir_order):
        B[2*k, :] = readout_z_fn.weight[k, :]

    for k in range(n_latents_in*iir_order):
        C[:, 2*k] = readout_H_fn.weight[:, k]

    # sys = signal.StateSpace(A.detach().cpu().numpy(), B.detach().cpu().numpy(), C.detach().cpu().numpy(), dt=20e-3)
    # zpk = sys.to_zpk()
    tf = ss2tf(A_np, B.detach().cpu().numpy()[::2] +0j, C.detach().cpu().numpy()[:, ::2]+0j, None)
    # tf = ss2tf(A.detach().cpu().numpy(), B.detach().cpu().numpy(), C.detach().cpu().numpy(), None)
    ir = torch.cat([(C @ A.matrix_power(t) @ B).unsqueeze(-1) for t in range(T_iir)], dim=-1)

    poles = np.roots(tf[1])
    zeros = [np.roots(tf_num) for tf_num in tf[0]]
    freqz = [scipy.signal.freqz(tf_num, tf[1], worN=1024) for tf_num in tf[0]]
    return poles, zeros, ir, freqz


def collate_model_parts(cfg, ssm):
    data = {}
    data['dynamics_z_K'] = ssm.dynamics_mod.mean_fn.transition_z_K

    transition_fn = ssm.dynamics_mod.mean_fn.transition_H_KK  # "A"
    readout_H_fn = ssm.dynamics_mod.mean_fn.readout_fn_H_KK   # "C"
    readout_z_fn = ssm.dynamics_mod.mean_fn.readout_fn_z_KK   # "B"
    freqz = [[[] for j in range(cfg.n_regions-1)] for i in range(cfg.n_regions)]
    poles = [[[] for j in range(cfg.n_regions-1)] for i in range(cfg.n_regions)]
    zeros = [[[] for j in range(cfg.n_regions-1)] for i in range(cfg.n_regions)]
    impulse = [[[] for j in range(cfg.n_regions-1)] for i in range(cfg.n_regions)]

    for k in range(cfg.n_regions):
        for l, l_tilde in zip(range(cfg.n_regions - 1), chain(range(k), range(k+1, cfg.n_regions))):
            poles[k][l], zeros[k][l], impulse[k][l], freqz[k][l] = convert_region_dynamics_ssm_ABC(
                cfg.n_latents_K[l_tilde], cfg.n_latents_K[k], transition_fn[k][l],readout_H_fn[k][l], readout_z_fn[k][l])

    data['poles'] = poles
    data['zeros'] = zeros
    data['impulse'] = impulse
    data['freqz'] = freqz

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(cfg.n_regions, cfg.n_regions - 1)
    if cfg.n_regions - 1 == 1:
        axs = axs.reshape(-1, 1)

    with torch.no_grad():
        for k in range(cfg.n_regions):
            for l, l_tilde in zip(range(cfg.n_regions - 1), chain(range(k), range(k + 1, cfg.n_regions))):
                freq_response = torch.fft.rfft(impulse[k][l], dim=2)
                fftfreq = torch.fft.rfftfreq(impulse[k][l].shape[-1])

                for i in range(freq_response.shape[0]):
                    for j in range(freq_response.shape[1]):
                        freq_response_dB = 20 * torch.log10(torch.abs(freq_response[i, j]))
                        # axs[k, l].plot(fftfreq[:freq_response_dB.shape[0]//2], freq_response_dB[:freq_response_dB.shape[0]//2])
                        axs[k, l].plot(fftfreq[:freq_response_dB.shape[0]//2], freq_response_dB[:freq_response_dB.shape[0]//2])
    fig.show()
    return data


def predict_trajectory(z_init, dynamics_fn, n_bins, impulse=None):
    z = [z_init.unsqueeze(1)]

    for t in range(n_bins-1):
        if impulse is None:
            z.append(dynamics_fn(z[-1]))
        elif impulse is not None and t != 100:
            z.append(dynamics_fn(z[-1]))
        elif impulse is not None and t == 100:
            z.append(dynamics_fn(z[-1]) + impulse)

    z = torch.cat(z, dim=1)
    return z


def gaussian_fa(y, n_components, model_device):
    n_trials, n_time_bins, n_neurons = y.shape
    y_tilde = rearrange(y, 'b t n -> (b t) n', n=n_neurons)

    fa = FactorAnalysis(n_components=n_components)

    with torch.no_grad():
        fa.fit(y_tilde.detach().cpu())

        C_hat = fa.components_.T
        C_hat = torch.tensor(C_hat, dtype=y.dtype, device=model_device)

        b_hat = fa.mean_
        b_hat = torch.tensor(b_hat, dtype=y.dtype, device=model_device)

        R_hat = fa.noise_variance_
        R_hat = torch.tensor(R_hat, dtype=y.dtype, device=model_device)

    params = {}
    params['R_hat'] = R_hat
    params['C_hat'] = C_hat
    params['b_hat'] = b_hat

    return params


def fit_pfa_model(bin_sz, n_obs, n_latents, y, device, batch_sz, n_epoch):
    n_trials, _, n_neurons = y.shape
    dataset = torch.utils.data.TensorDataset(y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_sz, shuffle=True)

    b_hat = prob_utils.estimate_poisson_rate_bias(y, bin_sz).clip(min=-5)
    y_nrm = y - bin_sz * torch.exp(b_hat)
    _, S, VmT = torch.linalg.svd(y_nrm.reshape(-1, n_neurons), full_matrices=False)

    pfa = PoissonFA(bin_sz, n_obs, n_latents, device)
    pfa.readout_fn.weight.data = VmT.mT[:, :n_latents] * S[:n_latents] / S[:n_latents].max()
    pfa.readout_fn.bias.data = b_hat
    opt = torch.optim.Adam(pfa.parameters(), lr=1e-3)
    z_hat = []

    for i in range(n_epoch):
        for batch in dataloader:
            opt.zero_grad()
            loss, stats = pfa(batch[0])
            loss += torch.linalg.svdvals(pfa.readout_fn.weight.data).pow(2).sum()
            loss.backward()
            opt.step()

            pfa.readout_fn.bias.data = pfa.readout_fn.bias.data.clip(min=-5, max=8)

            if i == n_epoch - 1:
                z_hat.append(stats['m'])

    z_hat = torch.cat(z_hat, dim=0).detach()
    return pfa, z_hat


def fit_dynamics_model(mean_fn, z, batch_sz, n_epoch, n_prd, epochs_save=None):
    saved_models = []
    dataset = torch.utils.data.TensorDataset(z)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_sz, shuffle=True)
    opt = torch.optim.Adam(mean_fn.parameters(), lr=1e-3)

    for i in range(n_epoch):
        for batch in dataloader:
            z_gt = extract_random_submatrix(batch[0], n_prd+1)
            opt.zero_grad()
            z_prd = []

            for n in range(n_prd):
                if n == 0:
                    z_prd.append(mean_fn(z_gt[:, 0])) # + 1e-1 * torch.randn_like(z_gt[:, 0]))
                else:
                    z_prd.append(mean_fn(z_prd[-1]))# + 1e-1 * torch.randn_like(z_gt[:, 0]))

            z_prd = torch.stack(z_prd, dim=1)
            loss = (z_gt[:, 1:] - z_prd).pow(2).mean()
            loss.backward()
            opt.step()

            if epochs_save is not None and i in epochs_save:
                saved_models.append(copy.deepcopy(mean_fn))

    return saved_models


def fit_dynamics_model_w_input(mean_fn, z, u, batch_sz, n_epoch, n_prd, epochs_save=None):
    saved_models = []
    n_latents = z.shape[-1]
    dataset = torch.utils.data.TensorDataset(z, u)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_sz, shuffle=True)
    opt = torch.optim.Adam(mean_fn.parameters(), lr=1e-3)

    for i in range(n_epoch):
        n_prd_i = min(i+1, n_prd)
        for batch in dataloader:
            z_u_b_cat = torch.cat((batch[0], batch[1]), dim=-1)
            z_u_b_gt = extract_random_submatrix(z_u_b_cat, n_prd_i+1)
            z_gt = z_u_b_gt[..., :n_latents]
            u_gt = z_u_b_gt[..., n_latents:]
            opt.zero_grad()
            z_prd = []
            # z_target = []

            for n in range(n_prd_i):
                if n == 0:
                    z_prd.append(mean_fn(z_gt[:, 0]) + u_gt[:, n]) # + 1e-1 * torch.randn_like(z_gt[:, 0]))
                else:
                    z_prd.append(mean_fn(z_prd[-1]) + u_gt[:, n])# + 1e-1 * torch.randn_like(z_gt[:, 0]))

            z_prd = torch.stack(z_prd, dim=1)
            loss = (z_gt[:, 1:] - z_prd).pow(2).mean()
            loss.backward()
            opt.step()

        if epochs_save is not None and i in epochs_save:
            saved_models.append(copy.deepcopy(mean_fn))

    return saved_models


def extract_random_submatrix(z: torch.Tensor, S: int) -> torch.Tensor:
    """
    Extracts a consecutive submatrix of size (BxSxL) from z of size (BxTxL).
    The starting index for each batch is randomly chosen from [0, T-S].

    Args:
        z (torch.Tensor): Input tensor of shape (B, T, L).
        S (int): Number of consecutive elements to extract (S < T).

    Returns:
        torch.Tensor: Extracted submatrix of shape (B, S, L).
    """
    B, T, L = z.shape
    start_indices = torch.randint(0, T - S + 1, (B,), device=z.device)  # Random start indices per batch
    batch_indices = torch.arange(B, device=z.device).unsqueeze(1)  # Shape (B, 1)
    time_indices = start_indices[:, None] + torch.arange(S, device=z.device)  # Shape (B, S)

    return z[batch_indices, time_indices]  # Shape (B, S, L)


class PoissonFA(torch.nn.Module):
    def __init__(self, bin_sz, n_obs, n_latents, device):
        super().__init__()
        self.n_obs = n_obs
        self.n_latents = n_latents

        self.bin_sz = bin_sz
        self.device = device

        self.readout_fn = nn.Linear(n_latents, n_obs, device=device)

    def forward(self, y):
        b = self.readout_fn.bias
        C = self.readout_fn.weight

        exp_b = self.bin_sz * torch.exp(b)
        P_inv = torch.eye(self.n_latents, device=self.device) + exp_b * C.mT @ C
        P_inv_chol = torch.linalg.cholesky(P_inv)
        m = chol_bmv_solve(P_inv_chol, bmv(C.mT, y - exp_b))
        P = torch.cholesky_inverse(P_inv_chol)

        diag_CPC = torch.diag(C @ P @ C.mT)
        log_rate_hat = self.readout_fn(m)
        ell = y * log_rate_hat - self.bin_sz * torch.exp(log_rate_hat + 0.5 * diag_CPC)
        tr = torch.einsum('...ii -> ...', P)
        kl = 0.5 * (tr + bip(m, m) + 2 * torch.sum(torch.log(torch.diag(P_inv_chol))))

        loss = kl - ell.sum(dim=-1)
        loss = loss.mean()
        stats = {'m': m, 'P': P}
        return loss, stats

class ReadoutLatentMask(torch.nn.Module):
    def __init__(self, n_latents, n_latents_read):
        super().__init__()

        self.n_latents = n_latents
        self.n_latents_read = n_latents_read

    def forward(self, z):
        return z[..., :self.n_latents_read]

    def get_matrix_repr(self, device):
        H = torch.zeros((self.n_latents_read, self.n_latents), device=device)
        H[torch.arange(self.n_latents_read), torch.arange(self.n_latents_read)] = 1.0
        return H


@torch.jit.export
class DynamicsGRU(torch.nn.Module):
    def __init__(self, hidden_dim, latent_dim, device, mode='mlp'):
        super(DynamicsGRU, self).__init__()
        self.mode = mode
        self.device = device
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # TODO: check
        if mode == 'gru':
            self.gru_cell = nn.GRUCell(0, hidden_dim, device=device).to(device)
            self.h_to_z = nn.Linear(hidden_dim, latent_dim, bias=False, device=device).to(device)
            self.z_to_h = nn.Linear(latent_dim, hidden_dim, bias=False, device=device).to(device)
        elif mode == 'mlp':
            self.h_to_z = nn.Linear(hidden_dim, latent_dim, device=device).to(device)
            self.z_to_h = nn.Linear(latent_dim, hidden_dim, device=device).to(device)

    def forward(self, z):
        if self.mode == 'gru':
            h_in = self.z_to_h(z)
            h_in_shape = list(h_in.shape)[:-1]
            h_in = h_in.reshape((-1, self.hidden_dim))

            empty_vec = torch.empty((h_in.shape[0], 0), device=z.device)
            h_out = self.gru_cell(empty_vec, h_in)
            h_out = h_out.reshape(h_in_shape + [self.hidden_dim])
            z_out = self.h_to_z(h_out)
            return z_out
        elif self.mode == 'mlp':
            h = nn.functional.relu(self.z_to_h(z))
            z_out = self.h_to_z(h)
            return z_out

    # def forward(self, z):
    #     h = nn.functional.sigmoid(self.z_to_h(z))
    #     z_out = self.h_to_z(h)
    #     return z_out


class FanInLinear(nn.Linear):
    # source: https://github.com/arsedler9/lfads-torch
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        # nn.init.constant_(self.bias, 0.0)

