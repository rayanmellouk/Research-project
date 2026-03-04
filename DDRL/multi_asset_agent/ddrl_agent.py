import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from DDRL.multi_asset_agent.market_env import MarketEnv
import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNet(nn.Module):
    def __init__(self, num_alphas, num_assets, hidden_dim=300, init_weights=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_alphas + num_assets, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_assets),
        )
        if init_weights:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Explicitly zero out the LAST layer to ensure initial actions are ~0
        # Access the last linear layer (index 4 in Sequential)
        last_layer = self.net[4]
        nn.init.uniform_(last_layer.weight, -1e-5, 1e-5)  # Near zero weights
        nn.init.constant_(last_layer.bias, 0)  # Zero bias

    def forward(self, state: torch.tensor):
        return self.net(state)


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
            num_alphas=env.num_alphas,
            num_assets=env.num_assets,
            hidden_dim=hidden_dim,
            init_weights=init_weights,
        ).to(device)
        self.optimizer = (
            optim.Adam(self.policy.parameters(), lr=lr)
            if optimizer_str == "adam"
            else optim.SGD(self.policy.parameters(), lr=lr)
        )
        self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            self.optimizer, lr_lambda=lambda epoch: 0.9
        )

    def rollout(self, U_batch: torch.tensor):
        batch_size = U_batch.shape[0]
        state = self.env.reset(batch_size)  # (batch,4)
        CR_t = torch.zeros(batch_size, device=self.device)

        for t in range(self.horizon):
            action = self.policy(state)
            CR_t = CR_t + self.env.reward(state, action)
            state = self.env.transition(state, action, U_batch[:, t, :])

        return CR_t

    def train_step(self, U_batch: torch.tensor):

        self.optimizer.zero_grad()
        
        cumulative_reward = self.rollout(U_batch)
        objective = cumulative_reward.mean()

        loss = -objective

        loss.backward()

        self.optimizer.step()

        return loss.item(), objective.item()
