import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# sys.path... n'est pas nécessaire DANS le fichier agent, mais gardez-le si besoin pour vos tests locaux
from DDRL.environments.market_env import MarketEnv
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNet(nn.Module):
    def __init__(self, state_dim=2, hidden_dim=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.tensor):
        return self.net(state)

class DDRLAgent:

    def __init__(
        self,
        env: MarketEnv,
        horizon: int = 50,
        hidden_dim: int = 300,
        lr: float = 1e-4, # On garde votre LR réduit
        device: str = "cpu",
        optimizer_str: str = "adam",
    ):
        self.env = env
        self.horizon = horizon
        self.device = device

        self.policy = PolicyNet(state_dim=2, hidden_dim=hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr) if optimizer_str == "adam" else optim.SGD(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lambda epoch: 0.9)

    def rollout(self, U_batch: torch.tensor, return_history: bool = False, debug: bool = False):
        """
        Exécute une trajectoire complète sur l'horizon T.
        """
        batch_size = U_batch.shape[0]
        state = self.env.reset(batch_size)  # (batch, 2)
        CR_t = torch.zeros(batch_size, device=self.device)

        # Variables pour le debug (moyennes sur l'horizon)
        total_signal = 0.0
        total_risk = 0.0
        total_cost = 0.0

        # Listes pour stocker l'historique si demandé
        states_list = []
        actions_list = []

        for t in range(self.horizon):
            action = self.policy(state)

            if return_history:
                states_list.append(state)
                actions_list.append(action)

            # Calcul des composantes pour le debug
            if debug:
                alpha_t, lw_t = state[:, 0:1], state[:, 1:2]
                # Note: On recalcule ici ce que fait env.reward pour l'affichage
                # Attention: assurez-vous que cela correspond EXACTEMENT à votre env.reward
                signal = (action * alpha_t).mean()
                risk = (0.5 * self.env.risk_lambda * action**2).mean()
                cost = (self.env.cost_C * (action - lw_t)**2).mean()
                
                total_signal += signal
                total_risk += risk
                total_cost += cost

            # Calcul de la reward réelle et transition
            CR_t = CR_t + self.env.reward(state, action)
            state = self.env.transition(state, action, U_batch[:, t, :])

        if debug:
            # On retourne un dictionnaire de stats moyennes par pas de temps
            debug_stats = {
                "signal": (total_signal / self.horizon).item(),
                "risk": (total_risk / self.horizon).item(),
                "cost": (total_cost / self.horizon).item()
            }
            return CR_t, debug_stats

        if return_history:
            states_history = torch.stack(states_list, dim=1)
            actions_history = torch.stack(actions_list, dim=1)
            return CR_t, states_history, actions_history

        return CR_t

    def train_step(self, U_batch: torch.tensor):
        self.optimizer.zero_grad()

        # On active le mode debug
        cumulative_reward, stats = self.rollout(U_batch, debug=True)
        objective = cumulative_reward.mean()

        loss = -objective
        
        # Affichage conditionnel (pour ne pas spammer, on peut le faire tout le temps ou 1 fois sur 100)
        # Ici on le fait tout le temps pour votre test immédiat
        #print(f"DEBUG | Signal: {stats['signal']:.4f} | Risk: {stats['risk']:.4f} | Cost: {stats['cost']:.4f} | Net: {objective.item()/self.horizon:.4f}")

        loss.backward()
        self.optimizer.step()

        return loss.item(), objective.item()
