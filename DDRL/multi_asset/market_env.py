# Implémentation de l'article DDRL dans le cas multi-actifs, inspiré de GP.

import numpy as np
import torch

class MarketEnv:
    def __init__(self, 
                 alpha_weights : torch.Tensor, 
                 omega : torch.Tensor, 
                 return_weights : torch.Tensor, 
                 sigma : torch.Tensor, 
                 trader_risk : torch.Tensor, 
                 dealer_risk : torch.Tensor, 
                 horizon : int, 
                 device : str = "cpu"):
        
        # ALPHA
        self.alpha_weights=alpha_weights
        self.omega=omega
        self.num_alphas = alpha_weights.shape[1]
        # ASSETS
        self.return_weights=return_weights
        self.sigma= sigma
        self.num_assets = return_weights.shape[1]
        # COST MATRIX UNDER ASSUMPTION A
        self.cost_lambda = trader_risk * sigma
        # MARKET PARAMETERS
        self.trader_risk=trader_risk
        self.dealer_risk=dealer_risk  
        self.horizon=horizon
        self.device=device
    
    def reset(self, batch_size):
        alpha_0 = torch.zeros(batch_size, self.num_alphas, device=self.device)
        l_w = torch.zeros(batch_size, self.num_assets, device=self.device)
        state = torch.cat([alpha_0, l_w], dim=-1) #(batch, num_alphas + num_assets)
        return state

    def transition(self, state, action, U):
        alpha_t = state[:,0:self.num_alphas] #(batch, num_alphas)
        lw_t = state[:,self.num_alphas:self.num_alphas+self.num_assets] #(batch, num_assets)

        L = torch.linalg.cholesky(self.omega) 
        alpha_next = alpha_t @ self.alpha_weights.T + U @ L.T #(batch, num_alphas)
        lw_next = action #(batch, 2)
        next_state = torch.cat([alpha_next, lw_next], dim = -1) #(batch, num_alphas + num_assets)
        return next_state

    def reward(self, state : torch.tensor, action : torch.tensor):
        alpha_t, lw_t = state[:,0:self.num_alphas], state[:,self.num_alphas:self.num_alphas+self.num_assets] #(batch, num_alphas + num_assets)
        w_t = action #(batch, num_assets)

        signal = torch.sum(
            w_t * (alpha_t @ self.return_weights.T),
            dim=1
        )  # (batch,)

        risk = 0.5 * self.trader_risk * torch.sum(
            w_t * (w_t @ self.sigma),
            dim=1
        )  # (batch,)   

        cost = torch.sum(
            (w_t - lw_t) * (self.cost_lambda @ (w_t - lw_t).T).T,
            dim=1
        )  # (batch,)

        r_t = signal - risk - cost
        return r_t  #(batch,)

    def generate_randomness(self, num_samples : int) :
        # Génère les variables U et V pour N trajectoires de longueur T
        U = torch.randn(num_samples, self.horizon, self.num_alphas, device=self.device)   #(batch, T, num_alphas)
        #V = torch.randn(num_samples, self.horizon, 2, device=self.device)   #(batch, T, 2), not used in this case. 
        return U #, V

