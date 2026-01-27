import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch 
from DDRL.multi_asset.market_env import MarketEnv
from DDRL.multi_asset.ddrl_agent import DDRLAgent
from DDRL.multi_asset.matrices import make_market_params
from tqdm import tqdm

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Market "hyperparameters"
    horizon = 50
    batch_size = 1024
    num_samples = int(1e6)
    steps_per_epoch = num_samples // batch_size
    n_epochs = 1
    num_assets = 2
    num_alphas = 2
    return_weights, alpha_weights, sigma, omega = make_market_params(num_assets, num_alphas, device)
    trader_risk = 1e-6
    dealer_risk = 1e-6

    # === Create environment and agent ===
    env = MarketEnv(
        alpha_weights=alpha_weights,
        omega=omega,
        return_weights=return_weights,
        sigma=sigma,
        trader_risk=trader_risk,
        dealer_risk=dealer_risk,
        horizon=horizon,
        device=device,
    )

    agent = DDRLAgent(
        env=env,
        horizon=horizon,
        hidden_dim=300,
        lr=1e-3,
        device=device,
    )

    # === Generate dataset of randomness for alphas ===
    print("Generating U dataset...")
    U_dataset = env.generate_randomness(num_samples)   # on CPU
    steps_per_epoch = num_samples // batch_size
    print(f"Dataset size: {num_samples}, batch_size: {batch_size}, "
          f"steps_per_epoch: {steps_per_epoch}")

    # === Training loop ===

    for epoch in range(n_epochs):
        indexes = torch.randperm(num_samples)
        print(f"Epoch {epoch + 1}/{n_epochs}")

        for it in tqdm(range(1, steps_per_epoch + 1), leave = False):
            indexes_batch = indexes[(it - 1) * batch_size : it * batch_size]
            U_batch = U_dataset[indexes_batch, :, :]  # (batch_size, horizon, 2)
            loss, avg_return = agent.train_step(U_batch)
        print(f"Epoch {epoch+1}: loss {loss}, avg_return {avg_return}")
        
        agent.scheduler.step()

    # === Optional: save trained policy ===
     # Dans trainer.py, à la fin
    checkpoint = {
        'model_state_dict': agent.policy.state_dict(),
        'market_params': {
            'alpha_weights': alpha_weights,
            'omega': omega,
            'return_weights': return_weights,  # C'est la matrice B importante
            'sigma': sigma,
            'trader_risk': trader_risk,
            'dealer_risk': dealer_risk,
            'horizon': horizon
        }
    }
    torch.save(checkpoint, "checkpoint_multiasset.pt")
    print("Modèle et paramètres de marché sauvegardés dans checkpoint_multiasset.pt")

    torch.save(agent.policy.state_dict(), "policy_multiasset_quadratic.pt")
    print("Training complete, policy saved to policy_multiasset_quadratic.pt")


if __name__ == "__main__":
    main()



