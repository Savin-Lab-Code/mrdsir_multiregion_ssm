import torch
import torch.nn as nn
import dev.linalg_utils as linalg_utils
import dev.ssm_modules.dynamics as dynamics

from dev.linalg_utils import bmv, bip, bop, chol_bmv_solve


class NonlinearFilter(nn.Module):
    def __init__(self, dynamics_mod, initial_c_pdf, device):
        super(NonlinearFilter, self).__init__()

        self.device = device
        self.dynamics_mod = dynamics_mod
        self.initial_c_pdf = initial_c_pdf

    def forward(self,
                k: torch.Tensor,
                K: torch.Tensor,
                n_samples: int,
                get_v: bool = False,
                get_kl: bool = False):

        # mask data, 0: data available, 1: data missing
        n_trials, n_time_bins, n_latents, rank = K.shape

        kl = []
        m_f = []
        z_f = []
        z_p = []
        stats = {}

        Q_diag = self.dynamics_mod.get_Q(k.device)
        Q_sqrt_diag = torch.sqrt(Q_diag)
        n_latents = Q_diag.shape[0]

        for t in range(n_time_bins):
            if t == 0:
                m_0 = torch.zeros((n_trials, n_latents), device=k.device)
                m_0 += self.initial_c_pdf.get_m_init(k.device)
                Q_0_diag = self.initial_c_pdf.get_Q_init()

                I_pl_triple_chol = torch.linalg.cholesky(torch.eye(K.shape[-1], device=k.device)
                                                         + (K[:, 0].mT * Q_0_diag) @ K[:, 0])
                g_t = k[:, t] - bmv(K[:, t],
                                     chol_bmv_solve(I_pl_triple_chol,
                                        bmv(K[:, t].mT, m_0 + Q_diag * k[:, t])))

                m_f_t = m_0 + Q_0_diag * g_t

                w_t = torch.randn([n_samples] + list(m_f_t.shape)[:-1] + [K.shape[-1]], device=k.device)
                z_p_t = Q_0_diag.sqrt() * torch.randn([n_samples] + list(m_f_t.shape), device=k.device)
                z_f_t = m_f_t + z_p_t - bmv(K[:, 0], chol_bmv_solve(I_pl_triple_chol, bmv(K[:, 0].mT, z_p_t) + w_t))
                phi = linalg_utils.triangular_inverse(I_pl_triple_chol).mT

                qp = bip(Q_0_diag * g_t, g_t)
                tr = -torch.diagonal(phi.mT @ (K[:, t].mT * Q_0_diag) @ K[:, t] @ phi, dim1=-2, dim2=-1).sum(dim=-1)
                logdet = 2 * torch.sum(torch.log(torch.diagonal(I_pl_triple_chol, dim1=-2, dim2=-1)), dim=-1)
                kl_t = 0.5 * (qp + tr + logdet)
                kl.append(kl_t)

            else:
                m_fn_z_tm1 = self.dynamics_mod.mean_fn.forward_packed(z_f[t - 1])

                I_pl_triple_chol = torch.linalg.cholesky(torch.eye(K.shape[-1], device=k.device)
                                                         + (K[:, t].mT * Q_diag) @ K[:, t])
                g_t = k[:, t] - bmv(K[:, t],
                                     chol_bmv_solve(I_pl_triple_chol,
                                        bmv(K[:, t].mT, m_fn_z_tm1 + Q_diag * k[:, t])))

                m_f_t = m_fn_z_tm1 + Q_diag * g_t

                w_t = torch.randn(list(m_f_t.shape)[:-1] + [K.shape[-1]], device=k.device)
                z_p_t = Q_sqrt_diag * torch.randn(list(m_f_t.shape), device=k.device)
                z_f_t = m_f_t + z_p_t - bmv(K[:, t], chol_bmv_solve(I_pl_triple_chol, bmv(K[:, t].mT, z_p_t) + w_t))
                phi = linalg_utils.triangular_inverse(I_pl_triple_chol).mT

                qp = bip(Q_diag * g_t, g_t)
                tr = -torch.diagonal(phi.mT @ (K[:, t].mT * Q_diag) @ K[:, t] @ phi, dim1=-2, dim2=-1).sum(dim=-1)
                logdet = 2 * torch.sum(torch.log(torch.diagonal(I_pl_triple_chol, dim1=-2, dim2=-1)), dim=-1)
                kl_t = 0.5 * (qp + tr + logdet)
                kl.append(kl_t.mean(dim=0))

                m_f_t = m_f_t.mean(dim=0)

            m_f.append(m_f_t)
            z_f.append(z_f_t)
            z_p.append(z_p_t)

        z_f = torch.stack(z_f, dim=2)
        stats['m_f'] = torch.stack(m_f, dim=1)
        stats['z_p'] = torch.stack(z_p, dim=2)

        if get_kl:
            stats['kl'] = torch.stack(kl, dim=1)

        return z_f, stats


class NonlinearFilterWithInput(nn.Module):
    def __init__(self, dynamics_mod, initial_c_pdf, device):
        super(NonlinearFilterWithInput, self).__init__()

        self.device = device
        self.dynamics_mod = dynamics_mod
        self.initial_c_pdf = initial_c_pdf

    def forward(self,
                u: torch.Tensor,
                k: torch.Tensor,
                K: torch.Tensor,
                n_samples: int,
                get_v: bool = False,
                get_kl: bool = False):

        # mask data, 0: data available, 1: data missing
        n_trials, n_time_bins, n_latents, rank = K.shape

        kl = []
        m_f = []
        z_f = []
        z_p = []
        stats = {}

        # TODO: fix
        Q_diag = self.dynamics_mod.get_Q(k.device)
        Q_sqrt_diag = torch.sqrt(Q_diag)

        u_end_dx = sum(self.dynamics_mod.L_K)
        n_latents = Q_diag.shape[0]

        for t in range(n_time_bins):
            if t == 0:
                m_0 = torch.zeros((n_trials, n_latents), device=u.device)
                m_0 += self.initial_c_pdf.get_m_init(k.device)
                m_0[..., : u_end_dx] += u[:, t]
                Q_0_diag = self.initial_c_pdf.get_Q_init()

                I_pl_triple_chol = torch.linalg.cholesky(torch.eye(K.shape[-1], device=k.device)
                                                         + (K[:, 0].mT * Q_0_diag) @ K[:, 0])
                g_t = k[:, t] - bmv(K[:, t],
                                     chol_bmv_solve(I_pl_triple_chol,
                                        bmv(K[:, t].mT, m_0 + Q_diag * k[:, t])))

                m_f_t = m_0 + Q_0_diag * g_t

                w_t = torch.randn([n_samples] + list(m_f_t.shape)[:-1] + [K.shape[-1]], device=k.device)
                z_p_t = Q_0_diag.sqrt() * torch.randn([n_samples] + list(m_f_t.shape), device=k.device)
                z_f_t = m_f_t + z_p_t - bmv(K[:, 0], chol_bmv_solve(I_pl_triple_chol, bmv(K[:, 0].mT, z_p_t) + w_t))
                phi = linalg_utils.triangular_inverse(I_pl_triple_chol).mT

                qp = bip(Q_0_diag * g_t, g_t)
                tr = -torch.diagonal(phi.mT @ (K[:, t].mT * Q_0_diag) @ K[:, t] @ phi, dim1=-2, dim2=-1).sum(dim=-1)
                logdet = 2 * torch.sum(torch.log(torch.diagonal(I_pl_triple_chol, dim1=-2, dim2=-1)), dim=-1)
                kl_t = 0.5 * (qp + tr + logdet)
                kl.append(kl_t)

            else:
                m_fn_z_tm1 = self.dynamics_mod.mean_fn.forward_packed(z_f[t - 1])
                m_fn_z_tm1[..., :u_end_dx] += u[:, t]

                I_pl_triple_chol = torch.linalg.cholesky(torch.eye(K.shape[-1], device=k.device)
                                                         + (K[:, t].mT * Q_diag) @ K[:, t])
                g_t = k[:, t] - bmv(K[:, t],
                                    chol_bmv_solve(I_pl_triple_chol,
                                                   bmv(K[:, t].mT, m_fn_z_tm1 + Q_diag * k[:, t])))

                m_f_t = m_fn_z_tm1 + Q_diag * g_t

                w_t = torch.randn(list(m_f_t.shape)[:-1] + [K.shape[-1]], device=k.device)
                z_p_t = Q_sqrt_diag * torch.randn(list(m_f_t.shape), device=k.device)
                z_f_t = m_f_t + z_p_t - bmv(K[:, t], chol_bmv_solve(I_pl_triple_chol, bmv(K[:, t].mT, z_p_t) + w_t))
                phi = linalg_utils.triangular_inverse(I_pl_triple_chol).mT

                qp = bip(Q_diag * g_t, g_t)
                tr = -torch.diagonal(phi.mT @ (K[:, t].mT * Q_diag) @ K[:, t] @ phi, dim1=-2, dim2=-1).sum(dim=-1)
                logdet = 2 * torch.sum(torch.log(torch.diagonal(I_pl_triple_chol, dim1=-2, dim2=-1)), dim=-1)
                kl_t = 0.5 * (qp + tr + logdet)
                kl.append(kl_t.mean(dim=0))

                m_f_t = m_f_t.mean(dim=0)

            m_f.append(m_f_t)
            z_f.append(z_f_t)
            z_p.append(z_p_t)

        z_f = torch.stack(z_f, dim=2)
        stats['m_f'] = torch.stack(m_f, dim=1)
        stats['z_p'] = torch.stack(z_p, dim=2)

        if get_kl:
            stats['kl'] = torch.stack(kl, dim=1)

        return z_f, stats


class LowRankNonlinearStateSpaceModel(nn.Module):
    def __init__(self, dynamics_mod, likelihood_pdf, initial_c_pdf, nl_filter, device='cpu'):
        super(LowRankNonlinearStateSpaceModel, self).__init__()

        self.device = device
        self.nl_filter = nl_filter
        self.dynamics_mod = dynamics_mod
        self.initial_c_pdf = initial_c_pdf
        self.likelihood_pdf = likelihood_pdf

    @torch.jit.export
    def forward(self,
                y,
                n_samples: int,
                p_mask_a: float = 0.0):

        z_s, stats = self.smooth_1_to_T(y, n_samples, p_mask_a=p_mask_a, get_kl=True)

        z_K_s, H_KK_s = dynamics.unpack_dense_multiregion_state(z_s,
                                                                self.dynamics_mod.L_K,
                                                                self.dynamics_mod.iir_order)

        ell = self.likelihood_pdf.get_ell(y, z_K_s).mean(dim=0)
        loss = stats['kl'] - ell
        loss = loss.sum(dim=-1).mean()

        z_s = {'z_K': z_K_s, 'H_KK': H_KK_s, 's_K': z_s}
        return loss, z_s, stats

    def smooth_1_to_T(self,
                      y,
                      n_samples: int,
                      p_mask_a: float=0.0,
                      get_kl: bool=False,
                      get_v: bool=False):

        n_regions = len(y)
        device = y[0].device

        n_trials, n_time_bins, _ = y[0].shape
        # t_mask_a = [torch.bernoulli((1 - p_mask_a) * torch.ones((n_trials, n_time_bins), device=device))
        #             for k in range(n_regions)]

        k_y, K_y = self.likelihood_pdf.get_local_update(y, p_mask_a=p_mask_a)
        # k_y = t_mask_a[..., None] * k_y
        # K_y = t_mask_a[..., None, None] * K_y
        # k_b, K_b = self.backward_encoder(k_y, K_y)

        # k = k_b + k_y
        # K = torch.concat([K_b, K_y], dim=-1)
        k = k_y
        K = K_y
        z_s, stats = self.nl_filter(k, K, n_samples, get_kl=get_kl, get_v=get_v)
        return z_s, stats

    def predict_forward(self,
                        z_tm1: torch.Tensor,
                        n_bins: int):

        z_forward = []
        Q_sqrt = torch.sqrt(self.dynamics_mod.get_Q(z_tm1.device))

        for t in range(n_bins):
            if t == 0:
                z_t = self.dynamics_mod.mean_fn.forward_packed(z_tm1) + Q_sqrt * torch.randn_like(z_tm1, device=z_tm1.device)
            else:
                z_t = self.dynamics_mod.mean_fn.forward_packed(z_forward[t-1]) + Q_sqrt * torch.randn_like(z_forward[t-1], device=z_tm1.device)

            z_forward.append(z_t)

        z_forward = torch.stack(z_forward, dim=2)
        return z_forward

    def filter_w_partial_observations(self, y, mask, n_samples):
        n_regions = len(y)
        device = y[0].device

        n_trials, n_time_bins, _ = y[0].shape

        k_y, K_y = self.likelihood_pdf.get_local_update_w_mask(y, mask)
        z_s, stats = self.nl_filter(k_y, K_y, n_samples)

        z_K_s, H_KK_s = dynamics.unpack_dense_multiregion_state(z_s,
                                                                self.dynamics_mod.L_K,
                                                                self.dynamics_mod.iir_order)
        z_s = {'z_K': z_K_s, 'H_KK': H_KK_s, 's_K': z_s}
        return z_s, stats


class LowRankNonlinearStateSpaceModelWithInput(nn.Module):
    def __init__(self, dynamics_mod, likelihood_pdf, initial_c_pdf, nl_filter, input_mod, device='cpu'):
        super(LowRankNonlinearStateSpaceModelWithInput, self).__init__()

        self.device = device
        self.nl_filter = nl_filter
        self.input_mod = input_mod
        self.dynamics_mod = dynamics_mod
        self.initial_c_pdf = initial_c_pdf
        self.likelihood_pdf = likelihood_pdf

    @torch.jit.export
    def forward(self,
                y,
                u_K,
                n_samples: int,
                p_mask_a: float = 0.0):

        u_lik = self.input_mod.get_likelihood_term(None, u_K)
        u_dyn = torch.cat(self.input_mod.get_dynamics_term(None, u_K), dim=-1)
        z_s, stats = self.smooth_1_to_T(y, u_lik, u_dyn, n_samples, p_mask_a=p_mask_a, get_kl=True)

        z_K_s, H_KK_s = dynamics.unpack_dense_multiregion_state(z_s,
                                                                self.dynamics_mod.L_K,
                                                                self.dynamics_mod.iir_order)

        ell = self.likelihood_pdf.get_ell(y, z_K_s, u_lik).mean(dim=0)
        loss = stats['kl'] - ell
        loss = loss.sum(dim=-1).mean()

        z_s = {'z_K': z_K_s, 'H_KK': H_KK_s, 's_K': z_s}
        return loss, z_s, stats

    def smooth_1_to_T(self,
                      y,
                      u_lik,
                      u_dyn,
                      n_samples: int,
                      p_mask_a: float=0.0,
                      get_kl: bool=False,
                      get_v: bool=False):

        device = y[0].device

        n_trials, n_time_bins, _ = y[0].shape
        # t_mask_a = torch.bernoulli((1 - p_mask_a) * torch.ones((n_trials, n_time_bins), device=device))

        # p_mask_a_K = len(y) * p_mask_a * K_rand / K_rand.sum()
        # p_mask_a_K = torch.clip(p_mask_a_K, max=0.99)

        # p_mask_a_K = p_mask_a * torch.rand(len(y))
        # k_y, K_y = self.likelihood_pdf.get_local_update(y, u_lik, p_mask_a=[p_mask_a for k in range(len(y))])
        # k_y, K_y = self.likelihood_pdf.get_local_update(y, u_lik, p_mask_a=[p_mask_a_k for p_mask_a_k in p_mask_a_K])
        k_y, K_y = self.likelihood_pdf.get_local_update(y, u_lik, p_mask_a=p_mask_a)
        # k_y = t_mask_a[..., None] * k_y
        # K_y = t_mask_a[..., None, None] * K_y
        # k_b, K_b = self.backward_encoder(k_y, K_y)

        # k = k_b + k_y
        # K = torch.concat([K_b, K_y], dim=-1)
        k = k_y
        K = K_y
        z_s, stats = self.nl_filter(u_dyn, k, K, n_samples, get_kl=get_kl, get_v=get_v)
        return z_s, stats

    def predict_forward(self,
                        z_tm1: torch.Tensor,
                        n_bins: int):

        z_forward = []
        Q_sqrt = torch.sqrt(self.dynamics_mod.get_Q(z_tm1.device))

        for t in range(n_bins):
            if t == 0:
                z_t = self.dynamics_mod.mean_fn(z_tm1) + Q_sqrt * torch.randn_like(z_tm1, device=z_tm1.device)
            else:
                z_t = self.dynamics_mod.mean_fn(z_forward[t-1]) + Q_sqrt * torch.randn_like(z_forward[t-1], device=z_tm1.device)

            z_forward.append(z_t)

        z_forward = torch.stack(z_forward, dim=2)
        return z_forward

    def predict_forward_w_input(self,
                                z_tm1: torch.Tensor,
                                u_KK,
                                n_bins: int,
                                use_noise=True):

        u_end_dx = sum(self.dynamics_mod.L_K)
        u_dyn = torch.cat(self.input_mod.get_dynamics_term(None, u_KK), dim=-1)

        z_forward = []
        Q_sqrt = torch.sqrt(self.dynamics_mod.get_Q(z_tm1.device))

        for t in range(n_bins):
            if t == 0:
                z_t = self.dynamics_mod.mean_fn.forward_packed(z_tm1)
            else:
                z_t = self.dynamics_mod.mean_fn.forward_packed(z_forward[t-1])

            if use_noise:
                z_t += Q_sqrt * torch.randn_like(z_tm1, device=z_tm1.device)

            z_t[..., :u_end_dx] += u_dyn[:, t]
            z_forward.append(z_t)

        z_forward = torch.stack(z_forward, dim=2)
        return z_forward
