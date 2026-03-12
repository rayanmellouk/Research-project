import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from DDRL.multi_asset_agent.market_env import MarketEnv
from DDRL.multi_asset_agent.matrices import unflatten_env_params_batch
import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNet(nn.Module):
    def __init__(self, state_dim, num_assets, num_alphas, hidden_dim=300, init_weights=True):
        super().__init__()
        self.num_alphas = num_alphas
        self.num_assets = num_assets
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_assets),
        )
        # Normalization buffers (identity until calibrated)
        zeta_len = state_dim - num_alphas - num_assets
        self.register_buffer('zeta_mean', torch.zeros(zeta_len))
        self.register_buffer('zeta_std', torch.ones(zeta_len))
        self.register_buffer('alpha_std', torch.ones(num_alphas))
        self.register_buffer('lw_std', torch.ones(num_assets))
        if init_weights:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Explicitly zero out the LAST layer to ensure initial actions are ~0
        # Access the last linear layer (index 4 in Sequential)
        last_layer = self.net[-1]
        nn.init.uniform_(last_layer.weight, -1e-5, 1e-5)  # Near zero weights
        nn.init.constant_(last_layer.bias, 0)  # Zero bias

    def _per_sample_scales(self, zeta):
        K, S = self.num_alphas, self.num_assets
        A, B, Sigma, L_omega, trader_risk, _ = unflatten_env_params_batch(zeta, S, K)

        rhos = torch.diagonal(A, dim1=-2, dim2=-1)
        Omega = L_omega @ L_omega.transpose(1, 2)
        omega_diag = torch.diagonal(Omega, dim1=-2, dim2=-1)
        alpha_std = torch.sqrt((omega_diag / (1.0 - rhos ** 2)).clamp_min(1e-12))

        denom = 1.0 - rhos.unsqueeze(2) * rhos.unsqueeze(1)
        Sigma_f = Omega / denom
        gamma_Sigma = trader_risk.unsqueeze(1).unsqueeze(2) * Sigma
        inv_gS_B = torch.linalg.solve(gamma_Sigma, B)
        pos_var = inv_gS_B @ Sigma_f @ inv_gS_B.transpose(1, 2)
        lw_std = torch.sqrt(torch.diagonal(pos_var, dim1=-2, dim2=-1).clamp_min(1e-12))

        return alpha_std.clamp_min(1e-8), lw_std.clamp_min(1e-8)

    def forward(self, state):
        K = self.num_alphas
        K_S = K + self.num_assets
        alpha = state[:, :K]
        lw = state[:, K:K_S]
        zeta = state[:, K_S:]

        # Per-sample scales, detached (no gradient through normalization)
        alpha_std, lw_std = self._per_sample_scales(zeta.detach())

        alpha_norm = alpha / alpha_std
        lw_norm = lw / lw_std
        zeta_norm = (zeta - self.zeta_mean) / self.zeta_std

        raw = self.net(torch.cat([alpha_norm, lw_norm, zeta_norm], dim=-1))
        return raw * lw_std  # rescale output back to position space


    def calibrate_zeta_norm(self, zeta_sample: torch.Tensor):
        """Set normalization stats from a batch of raw zeta vectors."""
        self.zeta_mean = zeta_sample.mean(dim=0)
        self.zeta_std = zeta_sample.std(dim=0).clamp_min(1e-8)

        K, S = self.num_alphas, self.num_assets
        A, B, Sigma, L_omega, trader_risk, _ = unflatten_env_params_batch(zeta_sample, S, K)

        rhos = torch.diagonal(A, dim1=-2, dim2=-1)
        Omega = L_omega @ L_omega.transpose(1, 2)
        omega_diag = torch.diagonal(Omega, dim1=-2, dim2=-1)
        alpha_std_batch = torch.sqrt((omega_diag / (1.0 - rhos ** 2)).clamp_min(1e-12))
        self.alpha_std = alpha_std_batch.mean(dim=0).clamp_min(1e-8)

        sigma_diag = torch.diagonal(Sigma, dim1=-2, dim2=-1)
        denom = 1.0 - rhos.unsqueeze(2) * rhos.unsqueeze(1)
        Sigma_f = Omega / denom
        pred_var = B @ Sigma_f @ B.transpose(1, 2)
        pred_std = torch.sqrt(torch.diagonal(pred_var, dim1=-2, dim2=-1).clamp_min(1e-12))
        lw_std_batch = (pred_std / (trader_risk.unsqueeze(1) * sigma_diag)).clamp_min(1e-8)
        self.lw_std = lw_std_batch.mean(dim=0).clamp_min(1e-8)


class DDRLAgent:

    def __init__(
        self,
        env: MarketEnv,
        horizon: int = 50,
        hidden_dim: int = 300,
        lr: float = 1e-3,
        device: str = "cpu",
        optimizer_str: str = "adam",
        init_weights: bool = False,
    ):
        self.env = env
        self.horizon = horizon
        self.device = device

        self.policy = PolicyNet(
            state_dim=env.state_dim,
            num_assets=env.num_assets,
            num_alphas=env.num_alphas,
            hidden_dim=hidden_dim,
            init_weights=init_weights,
        ).to(device)

        # Calibrate zeta normalization from a sample of the training distribution
        with torch.no_grad():
            calib_state = env.reset(4096)
            calib_zeta = calib_state[:, env.num_alphas + env.num_assets:]
            self.policy.calibrate_zeta_norm(calib_zeta)
        self.optimizer = (
            optim.Adam(self.policy.parameters(), lr=lr)
            if optimizer_str == "adam"
            else optim.SGD(self.policy.parameters(), lr=lr)
        )
        self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            self.optimizer, lr_lambda=lambda epoch: 0.9
        )

    def rollout(self, U_batch: torch.tensor, zeta=None):
        batch_size = U_batch.shape[0]
        state = self.env.reset(batch_size, zeta=zeta)
        CR_t = torch.zeros(batch_size, device=self.device)

        for t in range(self.horizon):
            action = self.policy(state)
            CR_t = CR_t + self.env.reward(state, action)
            state = self.env.transition(state, action, U_batch[:, t, :])

        return CR_t

    def train_step(self, U_batch: torch.tensor, zeta=None):

        self.optimizer.zero_grad()
        
        cumulative_reward = self.rollout(U_batch, zeta=zeta)
        objective = cumulative_reward.mean()

        loss = -objective

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item(), objective.item()
