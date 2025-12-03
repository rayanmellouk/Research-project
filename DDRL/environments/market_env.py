#Inclure différents modèles alphas
#et différents modèles de coûts et risques

import numpy as np
import torch

class MarketEnv:
    def __init__(self, rho_alpha, eta_alpha, risk_lambda, cost_C, horizon, device):
        # description of the market environment 
        self.rho_alpha = rho_alpha
        self.eta_alpha = eta_alpha
        self.risk_lambda = risk_lambda
        self.cost_C = cost_C
        self.horizon = horizon
        self.device = device

    def reset(self, batch_size):
        alpha_0 = torch.zeros(batch_size, 1, device=self.device)
        l_w = torch.zeros(batch_size, 1, device=self.device)
        state = torch.cat([alpha_0, l_w], dim=-1) #(batch, 2)
        return state

    def transition(self, state, action, U):
        alpha_t = state[:,0:1] #(batch, 1)
        lw_t = state[:,1:2] #(batch, 1)
        alpha_next = self.rho_alpha * alpha_t + self.eta_alpha * U #(batch, 1)
        lw_next = action #(batch, 1)
        next_state = torch.cat([alpha_next, lw_next], dim = -1) #(batch, 2)
        return next_state

    def reward(self, state : torch.tensor, action : torch.tensor):
        alpha_t, lw_t = state[:,0], state[:,1]

        signal = action*alpha_t
        risk = 0.5*self.risk_lambda*(action**2) #quadratic
        cost = self.cost_C*(torch.abs(action-lw_t)**2) #quadratic

        r_t = signal - risk - cost
        return r_t.squeeze(-1) #(batch,)
    

    def generate_randomness(self, batch_size):
        # Génère les variables U et V pour N trajectoires de longueur T
        U = torch.randn(batch_size, self.horizon, self.state_dim)   # Exemple avec bruit gaussien
        V = torch.randn(batch_size, self.horizon)
        return U, V

