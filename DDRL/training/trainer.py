import torch

class Trainer:

    def __init__(self, env, policy, N=10_000_000, T=50, epochs=50, batch_size=1024, lr=0.001, betas=(0.9, 0.999), gamma=0.9):
        """
        Initialisation du Trainer pour la procédure Deep Differentiable RL.
        :param env: instance de MarketEnv
        :param policy: instance du PolicyNetwork (PyTorch)
        :param N: nombre d’échantillons (episodes/samples)
        :param T: horizon (longueur trajectoire)
        :param epochs: nombre d’époques d’apprentissage
        :param batch_size: taille du mini-batch pour Adam
        :param lr: learning rate initial (Adam)
        :param betas: paramètres Adam (par défaut PyTorch)
        :param gamma: taux de décroissance du learning rate par epoch
        """
        self.env = env
        self.policy = policy
        self.N = N
        self.T = T
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.gamma)

    def CRT(s0, Ut, Vt, A, env):
        """
        Calcule la reward cumulée sur une trajectoire.
        s0 : état initial [batch, state_dim]
        Ut : [batch, T, state_dim]
        Vt : [batch, T]
        A  : politique (réseau de neurones)
        env: objet MarketEnv
        """
        R = 0
        state = s0
        T = Ut.shape[1]
        for t in range(T):
            action = A(state)
            R += env.reward(state, action, Vt[:, t])
            state = env.transition(state, action, Ut[:, t])
        return R
    
    def train(env, A, N, T, epochs, batch_size, gamma_init=0.001):
        optimizer = torch.optim.Adam(A.parameters(), lr=gamma_init, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        for epoch in range(epochs):
            U, V = env.generate_randomness(N, T)
            s0 = env.initialize(N)
            sum_loss = 0
            for i in range(0, N, batch_size):
                s0_batch = s0[i:i+batch_size]
                U_batch = U[i:i+batch_size]
                V_batch = V[i:i+batch_size]
                optimizer.zero_grad()
                reward = Trainer.CRT(s0_batch, U_batch, V_batch, A, env)
                loss = -reward.mean()
                loss.backward()
                optimizer.step()
                sum_loss += loss.item() * s0_batch.shape[0]
            scheduler.step()
            print(f"Epoch {epoch}: loss = {sum_loss / N}")


