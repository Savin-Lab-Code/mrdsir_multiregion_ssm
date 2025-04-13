import math
import torch
import torch.nn as nn
import dev.utils as utils
import torch.nn.functional as Fn
import dev.prob_utils as prob_utils

from sklearn.linear_model import Ridge
from dev.linalg_utils import bmv, bip, bop
from torcheval.metrics.functional import r2_score


class LinearReadoutFn(nn.Module):
    def __init__(self, n_neurons, n_latents, n_latents_read, use_bias=True, device='cpu'):
        super(LinearReadoutFn, self).__init__()

        self.n_neurons = n_neurons
        self.n_latents = n_latents
        self.n_latents_read = n_latents_read

        self.mask_fn = utils.ReadoutLatentMask(n_latents, n_latents_read)
        self.linear_fn = nn.Linear(n_latents_read, n_neurons, bias=use_bias, device=device)

    def forward(self, z):
        z_read = self.mask_fn(z)
        return self.linear_fn(z_read)

    def get_matrix_repr(self, device):
        C_read = self.linear_fn.weight
        C_mask = torch.zeros((self.n_neurons, self.n_latents - self.n_latents_read), device=device)
        C_total = torch.cat([C_read, C_mask], dim=-1)
        return C_total, torch.arange(self.n_latents_read, device=device)


class LocalGaussianLikelihood(nn.Module):
    def __init__(self, readout_fn_k, n_neurons, R_diag, device='cpu', fix_R=False):
        super(LocalGaussianLikelihood, self).__init__()

        self.n_neurons = n_neurons
        self.readout_fn = readout_fn_k

        if fix_R:
            self.log_R = utils.softplus_inv(R_diag)
        else:
            self.log_R = torch.nn.Parameter(utils.softplus_inv(R_diag))

    def get_ell(self, y, z_k, u_k=None, heldin_dx=None):
        mean = self.readout_fn(z_k)
        cov = Fn.softplus(self.log_R.to(y.device))

        if u_k is not None:
            mean += u_k

        if heldin_dx is not None:
            log_prob = -0.5 * ((y[..., heldin_dx] - mean[..., heldin_dx]) ** 2 / cov[heldin_dx] - torch.log(cov[heldin_dx]) - math.log(2 * math.pi))
        else:
            log_prob = -0.5 * ((y - mean)**2 / cov + torch.log(cov) + math.log(2 * math.pi))

        log_p_y = log_prob.sum(dim=-1)

        return log_p_y


class PoissonLikelihood(nn.Module):
    def __init__(self, readout_fn, n_neurons, delta, device='cpu', p_mask=0.0):
        super(PoissonLikelihood, self).__init__()
        self.delta = delta
        self.device = device
        self.n_neurons = n_neurons
        self.readout_fn = readout_fn

    def get_ell(self, y, z, u=None, reduce_neuron_dim=True):
        log_exp = math.log(self.delta) + self.readout_fn(z)  # C @ z
        log_p_y = -torch.nn.functional.poisson_nll_loss(log_exp, y, full=True, reduction='none')

        if reduce_neuron_dim:
            return log_p_y.sum(dim=-1)
        else:
            return log_p_y


class LocalMultiRegionLikelihood(nn.Module):
    def __init__(self, likelihood_K, n_latents_all, device='cpu'):
        super(LocalMultiRegionLikelihood, self).__init__()

        self.n_latents_all = n_latents_all
        self.likelihood_K = torch.nn.ModuleList(likelihood_K)

    def get_ell(self, y, z_K, u_K=None, heldin_dx=None):
        log_p_y = 0.0

        if heldin_dx is None:
            for k in range(len(self.likelihood_K)):
                if u_K is None:
                    log_p_y += self.likelihood_K[k].get_ell(y[k], z_K[k])
                else:
                    log_p_y += self.likelihood_K[k].get_ell(y[k], z_K[k], u_K[k])
        else:
            for k in range(len(self.likelihood_K)):
                if u_K is None:
                    log_p_y += self.likelihood_K[k].get_ell(y[k], z_K[k], heldin_dx=heldin_dx[k])
                else:
                    log_p_y += self.likelihood_K[k].get_ell(y[k], z_K[k], u_K[k], heldin_dx=heldin_dx[k])

        return log_p_y

    def get_mse(self, y, z_hat):
        r2_scores = []

        # for k in range(len(y)):
        for k in range(len(y)):
            y_hat_k = self.likelihood_K[k].readout_fn(z_hat[k]).mean(dim=0)
            # y_hat.append(y_hat_k.mean(dim=0))
            # r2_scores.append(r2_score(y[k].reshape(-1, y[k].shape[-1]), y_hat_k.reshape(-1, y[k].shape[-1])))
            with torch.no_grad():
                clf = Ridge(alpha=1e-1)

                try:
                    clf.fit(y_hat_k.reshape(-1, y[k].shape[-1]).cpu(), y[k].reshape(-1, y[k].shape[-1]).cpu())
                    score = clf.score(y_hat_k.reshape(-1, y[k].shape[-1]).cpu(), y[k].reshape(-1, y[k].shape[-1]).cpu())
                    r2_scores.append(score)
                except:
                    r2_scores.append(-1)

        # mse /= K
        # mse = sum(r2_scores) / len(r2_scores)
        return r2_scores

    def get_mse_gt(self, z_gt, z_hat):
        K = len(z_gt)
        r2_scores = []

        for k in range(K):
            with torch.no_grad():
                L_k = z_gt[k].shape[-1]
                clf = Ridge(alpha=1e-1)

                try:
                    clf.fit(z_hat[k].mean(dim=0).reshape(-1, L_k).cpu(),
                           z_gt[k].reshape(-1, L_k).cpu())
                    score = clf.score(z_hat[k].mean(dim=0).reshape(-1, L_k).cpu(),
                                      z_gt[k].reshape(-1, L_k).cpu())
                    r2_scores.append(score)
                except:
                    r2_scores.append(-1)

        return r2_scores

    def get_local_update(self, y_K, u_K=None, p_mask_a=0.):
        batch_sz = list(y_K[0].shape)[:-1]
        C_KK, b_K, read_dx_KK = self.get_readout_weights(y_K[0].device)
        n_latents_K = [C_KK[k].shape[1] for k in range(len(y_K))]

        if isinstance(p_mask_a, list):
            t_mask_a = [torch.bernoulli((1 - p_mask_a[k]) * torch.ones(batch_sz, device=y_K[0].device))
                        for k in range(len(y_K))]
        else:
            t_mask_a = [torch.bernoulli((1 - p_mask_a) * torch.ones(batch_sz, device=y_K[0].device))
                        for k in range(len(y_K))]

        k_y = torch.zeros(batch_sz + [self.n_latents_all], device=y_K[0].device)
        K_y = torch.zeros(batch_sz + [self.n_latents_all, sum(n_latents_K)], device=y_K[0].device)

        latents_k_sum = 0

        for k in range(len(y_K)):
            n_latents_k_end = n_latents_K[k] + latents_k_sum
            R = Fn.softplus(self.likelihood_K[k].log_R)
            R_inv = 1 / R

            if u_K is None:
                k_y_k = t_mask_a[k][..., None] * bmv(C_KK[k].mT, R_inv * (y_K[k] - b_K[k]))
            else:
                k_y_k = t_mask_a[k][..., None] * bmv(C_KK[k].mT, R_inv * (y_K[k] - b_K[k] - u_K[k]))

            k_y[..., latents_k_sum: n_latents_k_end] += k_y_k
            K_y_k = t_mask_a[k][..., None, None] * torch.linalg.cholesky((C_KK[k].mT * R_inv) @ C_KK[k])
            K_y[..., latents_k_sum: n_latents_k_end, latents_k_sum: n_latents_k_end] += K_y_k
            latents_k_sum += n_latents_K[k]

        return k_y, K_y

    def get_local_update_w_mask(self, y_K, mask_K, u_K=None, p_mask_a=0.):
        batch_sz = list(y_K[0].shape)[:-1]
        C_KK, b_K, read_dx_KK = self.get_readout_weights(y_K[0].device)
        n_latents_K = [C_KK[k].shape[1] for k in range(len(y_K))]

        if isinstance(p_mask_a, list):
            t_mask_a = [torch.bernoulli((1 - p_mask_a[k]) * torch.ones(batch_sz, device=y_K[0].device))
                        for k in range(len(y_K))]
        else:
            t_mask_a = [torch.bernoulli((1 - p_mask_a) * torch.ones(batch_sz, device=y_K[0].device))
                        for k in range(len(y_K))]

        k_y = torch.zeros(batch_sz + [self.n_latents_all], device=y_K[0].device)
        K_y = torch.zeros(batch_sz + [self.n_latents_all, sum(n_latents_K)], device=y_K[0].device)

        latents_k_sum = 0

        for k in range(len(y_K)):
            n_latents_k_end = n_latents_K[k] + latents_k_sum
            R = Fn.softplus(self.likelihood_K[k].log_R)
            R_inv = 1 / R

            if u_K is None:
                k_y_k = bmv(C_KK[k][mask_K[k], :].mT,
                            (R_inv * (y_K[k] - b_K[k]))[..., mask_K[k]])
            else:
                k_y_k = bmv(C_KK[k][mask_K[k], :].mT,
                            (R_inv * (y_K[k] - b_K[k] - u_K[k]))[..., mask_K[k]])

            k_y[..., latents_k_sum: n_latents_k_end] += k_y_k
            K_y_k = torch.linalg.cholesky((C_KK[k][mask_K[k], :].mT * R_inv[mask_K[k]]) @ C_KK[k][mask_K[k], :])
            K_y[..., latents_k_sum: n_latents_k_end, latents_k_sum: n_latents_k_end] += K_y_k
            latents_k_sum += n_latents_K[k]

        return k_y, K_y


    def get_readout_weights(self, device):
        b_K = []
        C_KK = []
        n_neurons = 0
        n_latents_K = 0

        read_dx_KK = []

        for likelihood_k in self.likelihood_K:
            b_k = likelihood_k.readout_fn.linear_fn.bias
            C_kk, read_dx_kk = likelihood_k.readout_fn.get_matrix_repr(device)
            read_dx_KK.append(read_dx_kk + n_latents_K)

            if b_k is None:
                b_k = torch.zeros(C_kk.shape[0], device=device)

            b_K.append(b_k)
            C_KK.append(C_kk)
            n_neurons += C_kk.shape[0]
            n_latents_K += C_kk.shape[1]

        return C_KK, b_K, read_dx_KK


class LocalMultiRegionPoissonLikelihood(nn.Module):
    def __init__(self, likelihood_K, encoder_K, n_latents_all, bin_sz, device='cpu'):
        super(LocalMultiRegionPoissonLikelihood, self).__init__()

        self.bin_sz = bin_sz
        self.n_latents_all = n_latents_all
        self.encoder_K = torch.nn.ModuleList(encoder_K)
        self.likelihood_K = torch.nn.ModuleList(likelihood_K)

    def get_ell(self, y, z_K, u_K=None):
        log_p_y = 0.0

        # for (y_k, z_k, likelihood_k) in zip(y, z_K, self.likelihood_K):
        for k in range(len(self.likelihood_K)):
            if u_K is None:
                log_p_y += self.likelihood_K[k].get_ell(y[k], z_K[k])
            else:
                log_p_y += self.likelihood_K[k].get_ell(y[k], z_K[k], u_K[k])

        return log_p_y

    def get_local_update(self, y_K, u_K=None, p_mask_a=0.):
        batch_sz = list(y_K[0].shape)[:-1]
        C_KK, b_K, read_dx_KK = self.get_readout_weights(y_K[0].device)
        n_latents_K = [C_KK[k].shape[1] for k in range(len(y_K))]

        if isinstance(p_mask_a, list):
            t_mask_a = [torch.bernoulli((1 - p_mask_a[k]) * torch.ones(batch_sz, device=y_K[0].device))
                        for k in range(len(y_K))]
        else:
            t_mask_a = [torch.bernoulli((1 - p_mask_a) * torch.ones(batch_sz, device=y_K[0].device))
                        for k in range(len(y_K))]

        k_y = torch.zeros(batch_sz + [self.n_latents_all], device=y_K[0].device)
        K_y = torch.zeros(batch_sz + [self.n_latents_all, sum(n_latents_K)], device=y_K[0].device)

        latents_k_sum = 0

        for k in range(len(y_K)):
            n_latents_k_end = n_latents_K[k] + latents_k_sum

            if u_K is None:
                R = self.bin_sz * torch.exp(b_K[k])
                # R = self.bin_sz * torch.exp(self.encoder_K[k](y_K[k]) + b_K[k])
                # R = (self.bin_sz * torch.exp(self.encoder_K[k](y_K[k]) + b_K[k])).clip(min=1e-3)
                k_y_k = t_mask_a[k][..., None] * bmv(C_KK[k].mT, (y_K[k] - R))
            else:
                R = self.bin_sz * torch.exp(b_K[k] + u_K[k])
                # R = self.bin_sz * torch.exp(b_K[k])
                # R = self.bin_sz * torch.exp(self.encoder_K[k](y_K[k]) + b_K[k] + u_K[k])
                # R = (self.bin_sz * torch.exp(self.encoder_K[k](y_K[k]) + b_K[k] + u_K[k])).clip(min=1e-3)
                k_y_k = t_mask_a[k][..., None] * bmv(C_KK[k].mT, (y_K[k] - R))

            k_y[..., latents_k_sum: n_latents_k_end] += k_y_k
            # print('---- torch.exp(b_K[k]) ----')
            # print(torch.exp(b_K[k]))
            # print('---- torch.exp(u_K[k]) ----')
            # print(torch.exp(u_K[k]))
            # print('---- torch.exp(b_K[k] + u_K[k]) ----')
            # print(torch.exp(b_K[k] + u_K[k]))
            # # try:
            K_y_k = t_mask_a[k][..., None, None] * torch.linalg.cholesky(C_KK[k].mT @ (R.unsqueeze(-1) * C_KK[k]))
            # except:
            #     print('here')
            K_y[..., latents_k_sum: n_latents_k_end, latents_k_sum: n_latents_k_end] += K_y_k
            latents_k_sum += n_latents_K[k]

        return k_y, K_y

    def get_readout_weights(self, device):
        b_K = []
        C_KK = []
        n_neurons = 0
        n_latents_K = 0

        read_dx_KK = []

        for likelihood_k in self.likelihood_K:
            b_k = likelihood_k.readout_fn.linear_fn.bias
            C_kk, read_dx_kk = likelihood_k.readout_fn.get_matrix_repr(device)
            read_dx_KK.append(read_dx_kk + n_latents_K)

            if b_k is None:
                b_k = torch.zeros(C_kk.shape[0], device=device)

            b_K.append(b_k)
            C_KK.append(C_kk)
            n_neurons += C_kk.shape[0]
            n_latents_K += C_kk.shape[1]

        return C_KK, b_K, read_dx_KK

    def get_mse(self, y, z_hat):
        bps = []
        K = len(y)

        for k in range(K):
            rate = self.bin_sz * torch.exp(self.likelihood_K[k].readout_fn(z_hat[k])).mean(dim=0)
            bps.append(prob_utils.bits_per_spike(torch.log(rate), y[k]))

        # bps /= K
        return bps
