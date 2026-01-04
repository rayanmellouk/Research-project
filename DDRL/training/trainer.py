import torch
from DDRL.environments.market_env import MarketEnv
from DDRL.agents.ddrl_agent import DDRLAgent
from tqdm import tqdm

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Hyperparameters ===
    horizon = 50
    batch_size = 1024
    num_samples = int(1e6)
    steps_per_epoch = num_samples // batch_size
    n_epochs = 50

    # Environment parameters (mono-scale alpha, quadratic risk & cost)
    rho_alpha = 0.9
    eta_alpha = 1.0
    risk_lambda = 1.0
    cost_C = 4.0

    # === Create environment and agent ===
    env = MarketEnv(
        rho_alpha=rho_alpha,
        eta_alpha=eta_alpha,
        risk_lambda=risk_lambda,
        cost_C=cost_C,
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

    # === Step 1: pre-generate all noise U for the dataset ===
    print("Generating U dataset...")
    U_dataset, _ = env.generate_randomness(num_samples)   # on CPU
    steps_per_epoch = num_samples // batch_size
    print(f"Dataset size: {num_samples}, batch_size: {batch_size}, "
          f"steps_per_epoch: {steps_per_epoch}")

    # === Training loop ===

    for epoch in range(n_epochs):
        indexes = torch.randperm(num_samples)
        print(f"Epoch {epoch + 1}/{n_epochs}")

        for it in tqdm(range(1, steps_per_epoch + 1)):
            indexes_batch = indexes[(it - 1) * batch_size : it * batch_size]
            U_batch = U_dataset[indexes_batch, :, :]  # (batch_size, horizon, 1)
            loss, avg_return = agent.train_step(U_batch)
        print(f"Epoch {epoch+1}: loss {loss}, avg_return {avg_return}")
        
        agent.scheduler.step()

    # === Optional: save trained policy ===
    torch.save(agent.policy.state_dict(), "policy_monoscale_quadratic.pt")
    print("Training complete, policy saved to policy_monoscale_quadratic.pt")


if __name__ == "__main__":
    main()



