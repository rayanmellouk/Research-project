# Implémentation de l'article DDRL dans le cas multi-actifs, inspiré de GP.
# Section 2.6: environment parameters ζ are part of the state.

import torch
from DDRL.multi_asset_agent.matrices import (
    zeta_dim, unflatten_env_params_batch, sample_env_params_batch
)

class MarketEnv:
    def __init__(self, num_alphas, num_assets, horizon, device="cpu", param_ranges=None):
        self.num_alphas = num_alphas
        self.num_assets = num_assets
        self.horizon = horizon
        self.device = device
        self.d_zeta = zeta_dim(num_assets, num_alphas)
        self.param_ranges = param_ranges or {}

    @property
    def state_dim(self):
        return self.num_alphas + self.num_assets + self.d_zeta

    def _split_state(self, state):
        K, S = self.num_alphas, self.num_assets
        alpha = state[:, :K]
        lw = state[:, K:K + S]
        zeta = state[:, K + S:]
        return alpha, lw, zeta
    
    def reset(self, batch_size, zeta=None):
        if zeta is None:
            zeta = sample_env_params_batch(
                batch_size, self.num_assets, self.num_alphas,
                self.device, **self.param_ranges
            ).detach() # The market parameters are static for each trajectory
        alpha_0 = torch.zeros(batch_size, self.num_alphas, device=self.device)
        lw_0 = torch.zeros(batch_size, self.num_assets, device=self.device)
        state = torch.cat([alpha_0, lw_0, zeta], dim=-1)
        return state

    def transition(self, state, action, U):
        alpha_t, lw_t, zeta = self._split_state(state)
        A, B, Sigma, L_omega, trader_risk, cost_lambda = unflatten_env_params_batch(
            zeta, self.num_assets, self.num_alphas
        )
        # Batched: alpha_next = alpha_t @ A^T + U @ L^T
        alpha_next = (torch.bmm(alpha_t.unsqueeze(1), A.transpose(1, 2)).squeeze(1)
                      + torch.bmm(U.unsqueeze(1), L_omega.transpose(1, 2)).squeeze(1))
        lw_next = action
        next_state = torch.cat([alpha_next, lw_next, zeta], dim=-1)
        return next_state

    def reward(self, state, action):
        alpha_t, lw_t, zeta = self._split_state(state)
        A, B, Sigma, L_omega, trader_risk, cost_lambda = unflatten_env_params_batch(
            zeta, self.num_assets, self.num_alphas
        )
        w_t = action

        # signal = w^T (B @ alpha)
        pred_return = torch.bmm(alpha_t.unsqueeze(1), B.transpose(1, 2)).squeeze(1)  # (batch, S)
        signal = torch.sum(w_t * pred_return, dim=1)  # (batch,)

        # risk = 0.5 * trader_risk * w^T Sigma w
        Sigma_w = torch.bmm(w_t.unsqueeze(1), Sigma).squeeze(1)  # (batch, S)
        risk = 0.5 * trader_risk * torch.sum(w_t * Sigma_w, dim=1)  # (batch,)

        # cost = 0.5 * dw^T Lambda dw
        dw = w_t - lw_t
        Lambda_dw = torch.bmm(dw.unsqueeze(1), cost_lambda).squeeze(1)  # (batch, S)
        cost = 0.5 * torch.sum(dw * Lambda_dw, dim=1)  # (batch,)

        return signal - risk - cost

    def generate_randomness(self, num_samples : int) :
        # Génère les variables U et V pour N trajectoires de longueur T
        U = torch.randn(num_samples, self.horizon, self.num_alphas, device=self.device)   #(batch, T, num_alphas)
        #V = torch.randn(num_samples, self.horizon, 2, device=self.device)   #(batch, T, 2), not used in this case. 
        return U #, V

