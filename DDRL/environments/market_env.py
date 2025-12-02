#Inclure différents modèles alphas
#et différents modèles de coûts et risques

import numpy as np
import torch

class MarketEnv:
    def __init__(self, phi, psi, state_dim, action_dim):
        # phi: fonction de transition d'état (doit être différentiable, ex. via PyTorch ou autograd)
        # psi: fonction de récompense (idem)
        self.phi = phi
        self.psi = psi
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = None

    def reset(self, initial_state):
        self.state = initial_state
        return self.state

    def step(self, action):
        ut = self.sample_ut() # Tirage de la variable aléatoire pour la transition
        next_state = self.phi(self.state, action, ut)
        vt = self.sample_vt() # Tirage pour la récompense
        reward = self.psi(self.state, action, vt)
        self.state = next_state
        return next_state, reward

    def sample_ut(self):
        # ex: return np.random.normal(loc=0, scale=1, size=self.state_dim)
        # adapter à ton setup !
        pass

    def sample_vt(self):
        # ex: return np.random.normal(loc=0, scale=1, size=1)
        # adapter à ton setup !
        pass
    

    def transition(self, state, action, U):
        # Fonction Φ : transition différentiable de l’état
        return self.phi(state, action, U)

    def reward(self, state, action, V):
        # Fonction Ψ : récompense différentiable
        return self.psi(state, action, V)

    def generate_randomness(self, N, T):
        # Génère les variables U et V pour N trajectoires de longueur T
        U = torch.randn(N, T, self.state_dim)   # Exemple avec bruit gaussien
        V = torch.randn(N, T)
        return U, V

    def initialize(self, N):
        # Fournit N états initiaux simulés
        return torch.zeros(N, self.state_dim)  # Exemple : tous zéros (adapter selon besoin)

