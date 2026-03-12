import sys
import os
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Adjust paths as per your original script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from DDRL.multi_asset_agent.market_env import MarketEnv
from DDRL.multi_asset_agent.ddrl_agent import DDRLAgent
from DDRL.multi_asset_agent.matrices import (
    flatten_env_params, sample_env_params_batch,
    unflatten_env_params_batch, save_env_params, load_env_params,
    zeta_from_json,
)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Market "hyperparameters"
    horizon = 50
    batch_size = 1024
    num_samples = int(1e6)
    n_epochs = 5
    num_assets = 2
    num_alphas = 2
    
    # === Create environment with variable params (Section 2.6) ===
    env = MarketEnv(
        num_alphas=num_alphas,
        num_assets=num_assets,
        horizon=horizon,
        device=device,
        param_ranges = dict(
            half_life_range = (10.0, 25.0),       # predictor persistence (days)
            asset_vol_range = (0.015, 0.03),      # daily asset volatility (1.5–3%)
            factor_vol_range = (0.8, 1.2),        # signal volatility scaling
            pred_std_frac = 0.5,                  # fraction of asset vol
            b_init_scale = 0.1,                   # initial signal loading
            trader_risk_range = (0.5, 1.0),       # risk aversion parameter
        ),
        cost_type="quadratic",  # or "linear"
    )

    agent = DDRLAgent(
        env=env,
        horizon=horizon,
        hidden_dim=300,
        lr=1e-3,
        device=device,
        init_weights=True,
    )
    
    # zeta_fixed = sample_env_params_batch(
    #     1, num_assets, num_alphas, device,
    #     **env.param_ranges
    # )  # (1, D_zeta) - one fixed environment
    # zeta_fixed = zeta_fixed.expand(batch_size, -1)  # (batch_size, D_zeta), all identical

    # === Generate dataset ===
    print("Generating U dataset...")
    # Note: Assuming env.generate_randomness exists or you handle it like in the snippet
    # If env.generate_randomness is not defined in the uploaded file, we generate manually:
    U_dataset = torch.randn(num_samples, horizon, num_alphas, device=device) # Assuming Gaussian noise
    
    steps_per_epoch = num_samples // batch_size

    # === Training loop ===
    for epoch in range(n_epochs):
        indexes = torch.randperm(num_samples)
        print(f"Epoch {epoch + 1}/{n_epochs}")
        total_return = 0.0
        running_loss = 0.0
        steps = 0
        pbar = tqdm(range(1, steps_per_epoch + 1), leave=False)
        
        for it in pbar:
            indexes_batch = indexes[(it - 1) * batch_size : it * batch_size]
            U_batch = U_dataset[indexes_batch, :, :] 

            loss, avg_return = agent.train_step(U_batch, zeta = None) # zeta_fixed can be used for sanity check 

            loss_val = (float(loss) if not torch.is_tensor(loss) else loss.detach().item())
            pbar.set_postfix(avg_return=f"{float(avg_return):.6f}")
            total_return += float(avg_return)
            running_loss += loss_val
            steps += 1
            
        avg_return_epoch = total_return / steps
        loss_epoch = running_loss / steps
        print(f"Epoch {epoch+1}: loss {loss_epoch:.8f}, avg_return {avg_return_epoch:.6f}")
        agent.scheduler.step()

    # === Save Checkpoint (model only — env params are variable) ===
    torch.save({"model_state_dict": agent.policy.state_dict()}, "norm_test_policy.pt")
    
    # ==========================================
    # === SANITY CHECK: evaluate on one fixed environment ===
    # ==========================================
    print("\n--- Sanity Check on a specific environment ---")

    # Sample one environment from the training distribution
    zeta = sample_env_params_batch(
        1, num_assets, num_alphas, device,
        **env.param_ranges
    )  # (1, D_zeta)

    # Recover raw matrices for saving
    A, B, Sigma, L_omega, trader_risk, cost_lambda = unflatten_env_params_batch(zeta, num_assets, num_alphas)
    Omega = L_omega @ L_omega.transpose(1, 2)

    # Save to JSON for reproducible re-evaluation later
    save_env_params(
        "debug_params.json",
        A[0], B[0], Sigma[0], Omega[0], trader_risk[0],
        num_assets, num_alphas, horizon,
    )
    print("Saved 'debug_params.json'")

    U_test = torch.randn(1, horizon, num_alphas, device=device)
    state = env.reset(1, zeta=zeta)
    trajectory_data = []

    with torch.no_grad():
        for t in range(horizon):
            current_alpha = state[0, :num_alphas].cpu().numpy()
            current_position = state[0, num_alphas:num_alphas + num_assets].cpu().numpy()

            action = agent.policy(state)
            next_position = action[0].cpu().numpy()

            row = {"t": t}
            for i in range(num_alphas):
                row[f"alpha_{i}"] = current_alpha[i]
            for i in range(num_assets):
                row[f"pos_prev_{i}"] = current_position[i]
            for i in range(num_assets):
                row[f"pos_new_{i}"] = next_position[i]

            trajectory_data.append(row)
            state = env.transition(state, action, U_test[:, t, :])

    df = pd.DataFrame(trajectory_data)
    df.to_csv("debug_variable_env.csv", index=False)
    print("Saved 'debug_variable_env.csv'")
    print("==========================================")

if __name__ == "__main__":
    main()