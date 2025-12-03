import torch
from DDRL.environments.market_env import MarketEnv
from DDRL.agents.ddrl_agent import DDRLAgent

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Hyperparameters ===
    horizon = 50
    batch_size = 256
    num_iterations = 1000

    # Environment parameters (mono-scale alpha, quadratic risk & cost)
    rho_alpha = 0.95
    eta_alpha = 0.1
    risk_lambda = 0.1
    cost_C = 0.1

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
        hidden_dim=64,
        lr=1e-3,
        device=device,
    )

    # === Training loop ===
    for it in range(1, num_iterations + 1):
        loss, avg_return = agent.train_step(batch_size)

        if it % 50 == 0:
            print(
                f"Iter {it:4d} | loss = {loss:8.4f} | "
                f"avg return = {avg_return:8.4f}"
            )

    # === Optional: save trained policy ===
    torch.save(agent.policy.state_dict(), "policy_monoscale_quadratic.pt")
    print("Training complete, policy saved to policy_monoscale_quadratic.pt")


if __name__ == "__main__":
    main()



