from DDRL.environments.market_env import MarketEnv
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNet(nn.Module) : 
    def __init__(self, state_dim =2 , hidden_dim =64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state : torch.tensor) : 
        return self.net(state)


class DDRLAgent :

    def __init__(
        self,
        env: MarketEnv,
        horizon: int = 50,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
    )   :
        self.env = env
        self.horizon = horizon
        self.device = device

        self.policy = PolicyNet(state_dim=2, hidden_dim=hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def rollout(self, batch_size : int) : 
        
        state = self.env.reset(batch_size) #(batch,2)
        CR_t = torch.zeros(batch_size, device = self.device)

        for _ in range(self.horizon) : 
            action = self.policy(state)
            CR_t = CR_t + self.env.reward(state, action)
            U = torch.randn(batch_size, 1, device=self.device)
            state = self.env.transition(state, action, U)
        
        return CR_t

    def train_step(self, batch_size) : 

        self.optimizer.zero_grad()

        cumulative_reward = self.rollout(batch_size)

        objective = cumulative_reward.mean()

        loss = -objective

        loss.backward()

        self.optimizer.step()

        return loss.item(), objective.item()