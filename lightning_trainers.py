import math
import time
import torch
import random
import dev.utils as utils
import matplotlib.pyplot as plt
import pytorch_lightning as lightning

from dev.ssm_modules.dynamics import (unpack_z_K,
                                      pack_dense_multiregion_state,
                                      unpack_dense_multiregion_state,
                                      stabilizeFcComplexDiagonalDynamics)

from dev.ssm_modules.likelihoods import LocalMultiRegionPoissonLikelihood



class LightningMultiRegionWithoutInput(lightning.LightningModule):
    def __init__(self, ssm, cfg, n_time_bins_ctx, is_svae=False, use_l1=False):
        super(LightningMultiRegionWithoutInput, self).__init__()

        self.ssm = ssm
        self.cfg = cfg
        self.use_l1 = use_l1
        self.is_svae = is_svae
        self.n_samples = cfg.n_samples

        self.p_mask_a = cfg.p_mask_a
        self.n_time_bins_ctx = n_time_bins_ctx
        self.save_hyperparameters(ignore=['ssm', 'cfg'])

    def training_step(self, batch, batch_idx):
        y = batch[:self.cfg.n_regions]
        y_ctx = [y_k[:, :] for y_k in y]
        y_prd = [y_k[:, self.n_time_bins_ctx:] for y_k in y]
        n_time_bins_prd = y_prd[0].shape[1]

        p_mask_a_t = self.p_mask_a * min(self.current_epoch / 5., 1.)
        # p_mask_a_t = [p_mask_a_t * (1 + math.cos(2 * math.pi * (self.current_epoch / (20+k) + k/len(y)))) / 2.0 for k in range(len(y))]
        p_mask_a_t = [p_mask_a_t * random.random() for k in range(len(y))]

        # b = [[], [], []]
        # for current_epoch in range(500):
        #     p_mask_a_t = [self.p_mask_a * (1 + math.cos(2 * math.pi * (current_epoch / (20+k) + k/len(y)))) / 2.0 for k in range(len(y))]
        #     b[0].append(p_mask_a_t[0])
        #     b[1].append(p_mask_a_t[1])
        #     b[2].append(p_mask_a_t[2])
        # plt.plot(b[0])
        # plt.plot(b[1])
        # plt.plot(b[2])
        # plt.show()

        t_start = time.time()
        loss, z_ctx, stats = self.ssm(y_ctx, self.n_samples, p_mask_a=p_mask_a_t)
        t_forward = time.time() - t_start

        z_K_reg_coeff = [5e-3, 5e-3, 5e-3]
        z_K_reg = [z_K_reg_coeff[k] * z_ctx['z_K'][k].shape[2] * z_ctx['z_K'][k].pow(2).sum(dim=-1).mean() for k in
                   range(self.cfg.n_regions)]
        loss += sum(z_K_reg)

        self.log("time_forward", t_forward, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        with torch.no_grad():
            z_ctx_packed = pack_dense_multiregion_state(z_ctx['z_K'], z_ctx['H_KK'])
            z_prd_packed = self.ssm.predict_forward(z_ctx_packed[:, :, self.n_time_bins_ctx],  n_time_bins_prd)
            z_K_prd, H_KK_prd = unpack_dense_multiregion_state(z_prd_packed, self.cfg.n_latents_K, self.cfg.iir_order)

            z_K_gt = batch[self.cfg.n_regions: 2 * self.cfg.n_regions]
            z_K_prd_gt = [z_K_gt_k[:, self.n_time_bins_ctx:] for z_K_gt_k in z_K_gt]

            mse_prd = self.ssm.likelihood_pdf.get_mse_gt(z_K_prd_gt, z_K_prd)
            mse_ctx = self.ssm.likelihood_pdf.get_mse_gt(z_K_gt, z_ctx['z_K'])

            for i in range(self.cfg.n_regions):
                # mse_prd = self.ssm.likelihood_pdf.get_mse([z_K_gt[i][:, self.n_time_bins_ctx:]], [z_K_prd[i]])
                # # mse_prd = self.ssm.likelihood_pdf.get_mse(y_prd, z_K_prd)
                # mse_ctx = self.ssm.likelihood_pdf.get_mse([z_K_gt[i]], [z_ctx['z_K'][i]])
                # # mse_ctx = self.ssm.likelihood_pdf.get_mse(y_ctx, z_ctx['z_K'])

                self.log(f"train_mse_{i}_ctx", mse_ctx[i], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"train_mse_{i}_prd", mse_prd[i], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            self.log(f"train_mse_ctx", sum(mse_ctx) / len(mse_ctx), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"train_mse_prd", sum(mse_prd) / len(mse_prd), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch[:self.cfg.n_regions]
        y_ctx = [y_k[:, :] for y_k in y]
        y_prd = [y_k[:, self.n_time_bins_ctx:] for y_k in y]
        n_time_bins_prd = y_prd[0].shape[1]

        loss, z_ctx, stats = self.ssm(y_ctx, self.n_samples)

        with torch.no_grad():
            z_ctx_packed = pack_dense_multiregion_state(z_ctx['z_K'], z_ctx['H_KK'])
            z_prd_packed = self.ssm.predict_forward(z_ctx_packed[:, :, self.n_time_bins_ctx], n_time_bins_prd)
            z_K_prd, H_KK_prd = unpack_dense_multiregion_state(z_prd_packed, self.cfg.n_latents_K, self.cfg.iir_order)

            z_K_gt = batch[self.cfg.n_regions: 2 * self.cfg.n_regions]
            z_K_prd_gt = [z_K_gt_k[:, self.n_time_bins_ctx:] for z_K_gt_k in z_K_gt]

            mse_prd = self.ssm.likelihood_pdf.get_mse_gt(z_K_prd_gt, z_K_prd)
            mse_ctx = self.ssm.likelihood_pdf.get_mse_gt(z_K_gt, z_ctx['z_K'])

            for i in range(self.cfg.n_regions):
                # mse_prd = self.ssm.likelihood_pdf.get_mse([z_K_gt[i][:, self.n_time_bins_ctx:]], [z_K_prd[i]])
                # # mse_prd = self.ssm.likelihood_pdf.get_mse(y_prd, z_K_prd)
                # mse_ctx = self.ssm.likelihood_pdf.get_mse([z_K_gt[i]], [z_ctx['z_K'][i]])
                # # mse_ctx = self.ssm.likelihood_pdf.get_mse(y_ctx, z_ctx['z_K'])

                self.log(f"valid_mse_{i}_ctx", mse_ctx[i], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"valid_mse_{i}_prd", mse_prd[i], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            self.log(f"valid_mse_ctx", sum(mse_ctx) / len(mse_ctx), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"valid_mse_prd", sum(mse_prd) / len(mse_prd), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            if self.current_epoch % 10 == 0:
                torch.save(self.ssm.state_dict(), f'results/state_dict_{self.logger.version}_{self.logger.name}.pt')

                fig, axs = plt.subplots(max(self.cfg.n_latents_K), self.cfg.n_regions, figsize=(3 * self.cfg.n_regions, 3))
                [axs[i, j].set_box_aspect(1.0) for i in range(max(self.cfg.n_latents_K)) for j in range(self.cfg.n_regions)]

                for k in range(self.cfg.n_regions):
                    [axs[i, k].plot(z_ctx['z_K'][k][s, 0, :, i].cpu(), alpha=0.7)
                     for i in range(self.cfg.n_latents_K[k]) for s in range(self.cfg.n_samples)]
                    axs[0, k].set_title(f'z[{k + 1}]')

                fig.savefig('plots/training/ssm_trajectories.pdf', bbox_inches='tight', transparent=True)
                plt.close()

        return loss

    def test_step(self, batch, batch_idx):
        y = batch[:self.cfg.n_regions]
        y_ctx = [y_k[:, :] for y_k in y]
        y_prd = [y_k[:, self.n_time_bins_ctx:] for y_k in y]
        n_time_bins_prd = y_prd[0].shape[1]

        loss, z_ctx, stats = self.ssm(y_ctx, self.n_samples)

        with torch.no_grad():
            z_ctx_packed = pack_dense_multiregion_state(z_ctx['z_K'], z_ctx['H_KK'])
            z_prd_packed = self.ssm.predict_forward(z_ctx_packed[:, :, self.n_time_bins_ctx],  n_time_bins_prd)
            z_K_prd, H_KK_prd = unpack_dense_multiregion_state(z_prd_packed, self.cfg.n_latents_K, self.cfg.iir_order)

            z_K_gt = batch[self.cfg.n_regions: 2 * self.cfg.n_regions]
            z_K_prd_gt = [z_K_gt_k[:, self.n_time_bins_ctx:] for z_K_gt_k in z_K_gt]

            mse_prd = self.ssm.likelihood_pdf.get_mse_gt(z_K_prd_gt, z_K_prd)
            mse_ctx = self.ssm.likelihood_pdf.get_mse_gt(z_K_gt, z_ctx['z_K'])

            for i in range(self.cfg.n_regions):
                # mse_prd = self.ssm.likelihood_pdf.get_mse([z_K_gt[i][:, self.n_time_bins_ctx:]], [z_K_prd[i]])
                # # mse_prd = self.ssm.likelihood_pdf.get_mse(y_prd, z_K_prd)
                # mse_ctx = self.ssm.likelihood_pdf.get_mse([z_K_gt[i]], [z_ctx['z_K'][i]])
                # # mse_ctx = self.ssm.likelihood_pdf.get_mse(y_ctx, z_ctx['z_K'])

                self.log(f"test_mse_{i}_ctx", mse_ctx[i], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"test_mse_{i}_prd", mse_prd[i], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            self.log(f"test_mse_ctx", sum(mse_ctx) / len(mse_ctx), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"test_mse_prd", sum(mse_prd) / len(mse_prd), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"test_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.ssm.parameters(), lr=self.cfg.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        log_Q_min = utils.softplus_inv(1e-3)
        log_Q_0_min = utils.softplus_inv(1e-1)

        with torch.no_grad():
            for k in range(self.cfg.n_regions):
                self.ssm.dynamics_mod.log_Q_K[k].data = torch.clip(self.ssm.dynamics_mod.log_Q_K[k].data, min=log_Q_min)
                self.ssm.initial_c_pdf.log_Q_K[k].data = torch.clip(self.ssm.initial_c_pdf.log_Q_K[k].data, min=log_Q_0_min)

                for l in range(self.cfg.n_regions-1):
                    self.ssm.initial_c_pdf.log_Q_H_KK[k][l].data = torch.clip(self.ssm.initial_c_pdf.log_Q_H_KK[k][l].data,
                                                                              min=log_Q_0_min)

            if self.is_svae:
                F = self.ssm.dynamics_mod.mean_fn.weight.data
                U, S, VmT = torch.linalg.svd(F, full_matrices=True)
                self.ssm.dynamics_mod.mean_fn.weight.data = (U * S.clip(max=1.0)) @ VmT

            stabilizeFcComplexDiagonalDynamics(self.ssm.dynamics_mod, angle_ub=math.pi / 3)


class LightningDirectlyConnectedWithoutInput(lightning.LightningModule):
    def __init__(self, ssm, cfg, n_time_bins_ctx, is_svae=False, use_l1=False):
        super(LightningDirectlyConnectedWithoutInput, self).__init__()

        self.ssm = ssm
        self.cfg = cfg
        self.use_l1 = use_l1
        self.is_svae = is_svae
        self.n_samples = cfg.n_samples

        self.p_mask_a = cfg.p_mask_a
        self.n_time_bins_ctx = n_time_bins_ctx
        self.save_hyperparameters(ignore=['ssm', 'cfg'])

    def training_step(self, batch, batch_idx):
        y = batch[:self.cfg.n_regions]
        y_ctx = [y_k[:, :] for y_k in y]
        y_prd = [y_k[:, self.n_time_bins_ctx:] for y_k in y]
        n_time_bins_prd = y_prd[0].shape[1]

        # p_mask_a_t = min(self.p_mask_a, self.current_epoch * self.p_mask_a / 1000)
        # p_mask_a_t = self.p_mask_a
        # p_mask_a_t = p_mask_a_t * (1 + math.cos(2 * math.pi * self.current_epoch / 20.)) / 2.0
        p_mask_a_t = self.p_mask_a
        # p_mask_a_t = min(self.p_mask_a, self.current_epoch * self.p_mask_a / 5)
        # p_mask_a_t = [p_mask_a_t * (1 + math.cos(2 * math.pi * (self.current_epoch / (20+k) + k/len(y)))) / 2.0 for k in range(len(y))]
        # p_mask_a_t = p_mask_a_t * (1 + math.cos(2 * math.pi * self.current_epoch / 20.)) / 2.0
        p_mask_a_t = [self.p_mask_a * random.random() for k in range(len(y))]

        t_start = time.time()
        loss, z_ctx, stats = self.ssm(y_ctx, self.n_samples, p_mask_a=p_mask_a_t)
        t_forward = time.time() - t_start

        z_K_reg_coeff = [5e-3, 5e-3, 5e-3]
        z_K_reg = [z_K_reg_coeff[k] * z_ctx['z_K'][k].shape[2] * z_ctx['z_K'][k].pow(2).sum(dim=-1).mean() for k in
                   range(self.cfg.n_regions)]
        loss += sum(z_K_reg)

        self.log("time_forward", t_forward, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        with torch.no_grad():
            z_ctx_packed = torch.cat(z_ctx['z_K'], dim=-1)
            z_prd_packed = self.ssm.predict_forward(z_ctx_packed[:, :, self.n_time_bins_ctx],  n_time_bins_prd)
            z_K_prd = unpack_z_K(z_prd_packed, self.cfg.n_latents_K)

            z_K_gt = batch[self.cfg.n_regions: 2 * self.cfg.n_regions]
            z_K_prd_gt = [z_K_gt_k[:, self.n_time_bins_ctx:] for z_K_gt_k in z_K_gt]

            mse_prd = self.ssm.likelihood_pdf.get_mse_gt(z_K_prd_gt, z_K_prd)
            mse_ctx = self.ssm.likelihood_pdf.get_mse_gt(z_K_gt, z_ctx['z_K'])

            for i in range(self.cfg.n_regions):
                # mse_prd = self.ssm.likelihood_pdf.get_mse([z_K_gt[i][:, self.n_time_bins_ctx:]], [z_K_prd[i]])
                # # mse_prd = self.ssm.likelihood_pdf.get_mse(y_prd, z_K_prd)
                # mse_ctx = self.ssm.likelihood_pdf.get_mse([z_K_gt[i]], [z_ctx['z_K'][i]])
                # # mse_ctx = self.ssm.likelihood_pdf.get_mse(y_ctx, z_ctx['z_K'])

                self.log(f"train_mse_{i}_ctx", mse_ctx[i], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"train_mse_{i}_prd", mse_prd[i], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            self.log(f"train_mse_ctx", sum(mse_ctx) / len(mse_ctx), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"train_mse_prd", sum(mse_prd) / len(mse_prd), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch[:self.cfg.n_regions]
        y_ctx = [y_k[:, :] for y_k in y]
        y_prd = [y_k[:, self.n_time_bins_ctx:] for y_k in y]
        n_time_bins_prd = y_prd[0].shape[1]

        loss, z_ctx, stats = self.ssm(y_ctx, self.n_samples)

        with torch.no_grad():
            z_ctx_packed = torch.cat(z_ctx['z_K'], dim=-1)
            z_prd_packed = self.ssm.predict_forward(z_ctx_packed[:, :, self.n_time_bins_ctx],  n_time_bins_prd)
            z_K_prd = unpack_z_K(z_prd_packed, self.cfg.n_latents_K)

            z_K_gt = batch[self.cfg.n_regions: 2 * self.cfg.n_regions]
            z_K_prd_gt = [z_K_gt_k[:, self.n_time_bins_ctx:] for z_K_gt_k in z_K_gt]

            mse_prd = self.ssm.likelihood_pdf.get_mse_gt(z_K_prd_gt, z_K_prd)
            mse_ctx = self.ssm.likelihood_pdf.get_mse_gt(z_K_gt, z_ctx['z_K'])

            for i in range(self.cfg.n_regions):
                # mse_prd = self.ssm.likelihood_pdf.get_mse([z_K_gt[i][:, self.n_time_bins_ctx:]], [z_K_prd[i]])
                # # mse_prd = self.ssm.likelihood_pdf.get_mse(y_prd, z_K_prd)
                # mse_ctx = self.ssm.likelihood_pdf.get_mse([z_K_gt[i]], [z_ctx['z_K'][i]])
                # # mse_ctx = self.ssm.likelihood_pdf.get_mse(y_ctx, z_ctx['z_K'])

                self.log(f"valid_mse_{i}_ctx", mse_ctx[i], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"valid_mse_{i}_prd", mse_prd[i], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            self.log(f"valid_mse_ctx", sum(mse_ctx) / len(mse_ctx), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"valid_mse_prd", sum(mse_prd) / len(mse_prd), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            if self.current_epoch % 10 == 0:
                torch.save(self.ssm.state_dict(), f'results/state_dict_{self.logger.version}_{self.logger.name}.pt')

                fig, axs = plt.subplots(max(self.cfg.n_latents_K), self.cfg.n_regions, figsize=(3 * self.cfg.n_regions, 3))
                [axs[i, j].set_box_aspect(1.0) for i in range(max(self.cfg.n_latents_K)) for j in range(self.cfg.n_regions)]

                for k in range(self.cfg.n_regions):
                    [axs[i, k].plot(z_ctx['z_K'][k][s, 0, :, i].cpu(), alpha=0.7)
                     for i in range(self.cfg.n_latents_K[k]) for s in range(self.cfg.n_samples)]
                    axs[0, k].set_title(f'z[{k + 1}]')

                fig.savefig('plots/training/ssm_trajectories.pdf', bbox_inches='tight', transparent=True)
                plt.close()

        return loss

    def test_step(self, batch, batch_idx):
        y = batch[:self.cfg.n_regions]
        y_ctx = [y_k[:, :] for y_k in y]
        y_prd = [y_k[:, self.n_time_bins_ctx:] for y_k in y]
        n_time_bins_prd = y_prd[0].shape[1]

        loss, z_ctx, stats = self.ssm(y_ctx, self.n_samples)

        with torch.no_grad():
            z_ctx_packed = torch.cat(z_ctx['z_K'], dim=-1)
            z_prd_packed = self.ssm.predict_forward(z_ctx_packed[:, :, self.n_time_bins_ctx],  n_time_bins_prd)
            z_K_prd = unpack_z_K(z_prd_packed, self.cfg.n_latents_K)

            z_K_gt = batch[self.cfg.n_regions: 2 * self.cfg.n_regions]
            z_K_prd_gt = [z_K_gt_k[:, self.n_time_bins_ctx:] for z_K_gt_k in z_K_gt]

            mse_prd = self.ssm.likelihood_pdf.get_mse_gt(z_K_prd_gt, z_K_prd)
            mse_ctx = self.ssm.likelihood_pdf.get_mse_gt(z_K_gt, z_ctx['z_K'])

            for i in range(self.cfg.n_regions):
                # mse_prd = self.ssm.likelihood_pdf.get_mse([z_K_gt[i][:, self.n_time_bins_ctx:]], [z_K_prd[i]])
                # # mse_prd = self.ssm.likelihood_pdf.get_mse(y_prd, z_K_prd)
                # mse_ctx = self.ssm.likelihood_pdf.get_mse([z_K_gt[i]], [z_ctx['z_K'][i]])
                # # mse_ctx = self.ssm.likelihood_pdf.get_mse(y_ctx, z_ctx['z_K'])

                self.log(f"test_mse_{i}_ctx", mse_ctx[i], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"test_mse_{i}_prd", mse_prd[i], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            self.log(f"test_mse_ctx", sum(mse_ctx) / len(mse_ctx), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"test_mse_prd", sum(mse_prd) / len(mse_prd), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.ssm.parameters(), lr=self.cfg.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        log_Q_min = utils.softplus_inv(1e-3)
        log_Q_0_min = utils.softplus_inv(1e-1)

        with torch.no_grad():
            for k in range(self.cfg.n_regions):
                self.ssm.dynamics_mod.log_Q_K[k].data = torch.clip(self.ssm.dynamics_mod.log_Q_K[k].data, min=log_Q_min)
                self.ssm.initial_c_pdf.log_Q_K[k].data = torch.clip(self.ssm.initial_c_pdf.log_Q_K[k].data, min=log_Q_0_min)


class LightningMultiRegionWithInput(lightning.LightningModule):
    def __init__(self, ssm, cfg, n_time_bins_ctx, is_svae=False, use_l1=False):
        super(LightningMultiRegionWithInput, self).__init__()

        self.ssm = ssm
        self.cfg = cfg
        self.use_l1 = use_l1
        self.is_svae = is_svae
        self.n_samples = cfg.n_samples

        self.p_mask_a = cfg.p_mask_a
        self.n_time_bins_ctx = n_time_bins_ctx
        self.save_hyperparameters(ignore=['ssm', 'cfg'])

    def training_step(self, batch, batch_idx):
        y = batch[:self.cfg.n_regions]
        u = batch[self.cfg.n_regions: 2*self.cfg.n_regions]

        y_ctx = [y_k[:, :] for y_k in y]
        u_ctx = [u_k[:, :] for u_k in u]
        y_prd = [y_k[:, self.n_time_bins_ctx:] for y_k in y]
        u_prd = [u_k[:, self.n_time_bins_ctx:] for u_k in u]
        n_time_bins_prd = y_prd[0].shape[1]

        # p_mask_a_t = self.p_mask_a
        # p_mask_a_t = min(self.p_mask_a, self.current_epoch * self.p_mask_a / 5)
        # p_mask_a_t = p_mask_a_t * (1 + math.cos(2 * math.pi * self.current_epoch / 20.)) / 2.0
        # p_mask_a_t = [p_mask_a_t * (1 + math.cos(2 * math.pi * (self.current_epoch / (20+k) + k/len(y)))) / 2.0 for k in range(len(y))]
        # p_mask_a_t = self.p_mask_a
        p_mask_a_t = self.p_mask_a * min(self.current_epoch / 5., 1.)
        p_mask_a_t = [p_mask_a_t * random.random() for k in range(len(y))]

        t_start = time.time()
        loss, z_ctx, stats = self.ssm(y_ctx, u_ctx, self.n_samples, p_mask_a=p_mask_a_t)
        t_forward = time.time() - t_start

        # z_K_reg_coeff = [5e-3, 5e-3, 5e-3]
        z_K_reg = [self.cfg.z_l2[k] * z_ctx['z_K'][k].pow(2).sum(dim=[-1, -2]).mean()
                   for k in range(self.cfg.n_regions)]
        loss += sum(z_K_reg)

        # for k in range(self.cfg.n_regions):
        #     s_vals = torch.linalg.svdvals(self.ssm.likelihood_pdf.likelihood_K[k].readout_fn.linear_fn.weight)
        #     loss += 1e-1 * s_vals.pow(2).sum()

        # if self.use_l1:
        #     readout_H_fn = self.ssm.dynamics_mod.mean_fn.readout_fn_H_KK    # "C"
        #     readout_z_fn = self.ssm.dynamics_mod.mean_fn.readout_fn_z_KK    # "B"
        #     for k in range(self.cfg.n_regions):
        #         for l in range(self.cfg.n_regions - 1):
        #             B, C = utils.ssm_get_BC(self.cfg.n_latents_K[l], self.cfg.n_latents_K[k], self.cfg.iir_order,
        #                                     readout_H_fn[k][l], readout_z_fn[k][l])
        #             loss += self.cfg.l1_penalty * torch.linalg.svdvals(C @ B).pow(2).sum().sqrt()

        self.log("time_forward", t_forward, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        with torch.no_grad():
            z_ctx_packed = pack_dense_multiregion_state(z_ctx['z_K'], z_ctx['H_KK'])
            z_prd_packed = self.ssm.predict_forward_w_input(z_ctx_packed[:, :, self.n_time_bins_ctx], u_prd,  n_time_bins_prd)
            z_K_prd, H_KK_prd = unpack_dense_multiregion_state(z_prd_packed, self.cfg.n_latents_K, self.cfg.iir_order)

            mse_prd = self.ssm.likelihood_pdf.get_mse(y_prd, z_K_prd)
            mse_ctx = self.ssm.likelihood_pdf.get_mse(y_ctx, z_ctx['z_K'])

            for i in range(self.cfg.n_regions):
                self.log(f"train_mse_{i}_ctx", mse_ctx[i], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"train_mse_{i}_prd", mse_prd[i], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.current_epoch % 10 == 0:
            y = batch[:self.cfg.n_regions]
            u = batch[self.cfg.n_regions: 2 * self.cfg.n_regions]
            y_ctx = [y_k[:, :] for y_k in y]
            u_ctx = [u_k[:, :] for u_k in u]
            y_prd = [y_k[:, self.n_time_bins_ctx:] for y_k in y]
            u_prd = [u_k[:, self.n_time_bins_ctx:] for u_k in u]
            n_time_bins_prd = y_prd[0].shape[1]

            loss, z_ctx, stats = self.ssm(y_ctx, u_ctx, self.n_samples)

            with torch.no_grad():
                z_ctx_packed = pack_dense_multiregion_state(z_ctx['z_K'], z_ctx['H_KK'])
                z_prd_packed = self.ssm.predict_forward_w_input(z_ctx_packed[:, :, self.n_time_bins_ctx], u_prd,  n_time_bins_prd)
                z_K_prd, H_KK_prd = unpack_dense_multiregion_state(z_prd_packed, self.cfg.n_latents_K, self.cfg.iir_order)

                mse_prd = self.ssm.likelihood_pdf.get_mse(y_prd, z_K_prd)
                mse_ctx = self.ssm.likelihood_pdf.get_mse(y_ctx, z_ctx['z_K'])

                for i in range(self.cfg.n_regions):
                    self.log(f"valid_mse_{i}_ctx", mse_ctx[i], on_step=False, on_epoch=True, prog_bar=True,
                             sync_dist=True)
                    self.log(f"valid_mse_{i}_prd", mse_prd[i], on_step=False, on_epoch=True, prog_bar=True,
                             sync_dist=True)

                self.log(f"valid_mse_ctx", sum(mse_ctx) / len(mse_ctx), on_step=False, on_epoch=True, prog_bar=True,
                         sync_dist=True)
                self.log(f"valid_mse_prd", sum(mse_prd) / len(mse_prd), on_step=False, on_epoch=True, prog_bar=True,
                         sync_dist=True)
                self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

                torch.save(self.ssm.state_dict(), f'results/state_dict_{self.logger.version}_{self.logger.name}.pt')

                fig, axs = plt.subplots(max(self.cfg.n_latents_K), self.cfg.n_regions, figsize=(3 * self.cfg.n_regions, 3))
                [axs[i, j].set_box_aspect(1.0) for i in range(max(self.cfg.n_latents_K)) for j in range(self.cfg.n_regions)]

                for k in range(self.cfg.n_regions):
                    [axs[i, k].plot(z_ctx['z_K'][k][s, 0, :, i].cpu(), alpha=0.7)
                     for i in range(self.cfg.n_latents_K[k]) for s in range(self.cfg.n_samples)]
                    axs[0, k].set_title(f'z[{k + 1}]')

                    fig.savefig('plots/training/ssm_trajectories.pdf', bbox_inches='tight', transparent=True)
                    plt.close()

            return loss
        return None

    def test_step(self, batch, batch_idx):
        y = batch[:self.cfg.n_regions]
        u = batch[self.cfg.n_regions: 2*self.cfg.n_regions]
        y_ctx = [y_k[:, :] for y_k in y]
        u_ctx = [u_k[:, :] for u_k in u]
        y_prd = [y_k[:, self.n_time_bins_ctx:] for y_k in y]
        u_prd = [u_k[:, self.n_time_bins_ctx:] for u_k in u]
        n_time_bins_prd = y_prd[0].shape[1]

        loss, z_ctx, stats = self.ssm(y_ctx, u_ctx, self.n_samples)

        with torch.no_grad():
            z_ctx_packed = pack_dense_multiregion_state(z_ctx['z_K'], z_ctx['H_KK'])
            z_prd_packed = self.ssm.predict_forward_w_input(z_ctx_packed[:, :, self.n_time_bins_ctx], u_prd,  n_time_bins_prd)
            z_K_prd, H_KK_prd = unpack_dense_multiregion_state(z_prd_packed, self.cfg.n_latents_K, self.cfg.iir_order)

            mse_prd = self.ssm.likelihood_pdf.get_mse(y_prd, z_K_prd)
            mse_ctx = self.ssm.likelihood_pdf.get_mse(y_ctx, z_ctx['z_K'])

            for i in range(self.cfg.n_regions):
                self.log(f"test_mse_{i}_ctx", mse_ctx[i], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"test_mse_{i}_prd", mse_prd[i], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            self.log(f"test_mse_ctx", sum(mse_ctx) / len(mse_ctx), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"test_mse_prd", sum(mse_prd) / len(mse_prd), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.ssm.parameters(), lr=self.cfg.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        log_Q_min = utils.softplus_inv(1e-3)
        log_Q_0_min = utils.softplus_inv(1e-1)

        with torch.no_grad():
            for k in range(self.cfg.n_regions):
                self.ssm.dynamics_mod.log_Q_K[k].data = torch.clip(self.ssm.dynamics_mod.log_Q_K[k].data, min=log_Q_min)
                self.ssm.initial_c_pdf.log_Q_K[k].data = torch.clip(self.ssm.initial_c_pdf.log_Q_K[k].data, min=log_Q_0_min)

                for l in range(self.cfg.n_regions-1):
                    self.ssm.initial_c_pdf.log_Q_H_KK[k][l].data = torch.clip(self.ssm.initial_c_pdf.log_Q_H_KK[k][l].data,
                                                                              min=log_Q_0_min)

            # -- keep SVAE stable
            if self.is_svae:
                F = self.ssm.dynamics_mod.mean_fn.weight.data
                U, S, VmT = torch.linalg.svd(F, full_matrices=True)
                self.ssm.dynamics_mod.mean_fn.weight.data = (U * S.clip(max=1.0)) @ VmT

            # -- help avoid degeneracy for low firing rate neurons for Poisson GLM
            if isinstance(self.ssm.likelihood_pdf, LocalMultiRegionPoissonLikelihood):
                for likelihood_k in self.ssm.likelihood_pdf.likelihood_K:
                    likelihood_k.readout_fn.linear_fn.bias.data = likelihood_k.readout_fn.linear_fn.bias.data.clip(min=-5)
                    likelihood_k.readout_fn.linear_fn.bias.data = likelihood_k.readout_fn.linear_fn.bias.data.clip(max=7)

        stabilizeFcComplexDiagonalDynamics(self.ssm.dynamics_mod, angle_ub=math.pi / 3)
