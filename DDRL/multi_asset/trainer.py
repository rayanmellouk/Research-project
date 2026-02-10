import sys
import os
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Adjust paths as per your original script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from DDRL.multi_asset.market_env import MarketEnv
from DDRL.multi_asset.ddrl_agent import DDRLAgent
from DDRL.multi_asset.matrices import make_market_params

def tensor_to_list(tensor):
    """Helper to convert tensors to standard python lists for JSON serialization"""
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy().tolist()
    return tensor

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Market "hyperparameters"
    horizon = 50
    batch_size = 1024
    num_samples = int(1e6)
    n_epochs = 1
    num_assets = 2
    num_alphas = 2
    
    # Generate Matrices
    return_weights, alpha_weights, sigma, omega = make_market_params(
        num_assets, num_alphas, device
    )   # Les alpha weights sont la matrice diag des Phi et les return_weights sont la matrice B, sigma est la matrice de covariance des retours et omega celle des alphas
    trader_risk = 0.5
    dealer_risk = 0.5

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
        lr=1e-4,
        device=device,
        init_weights=True,
    )
    
    # ... [Print statements omitted for brevity] ...

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

            loss, avg_return = agent.train_step(U_batch)

            loss_val = (float(loss) if not torch.is_tensor(loss) else loss.detach().item())
            pbar.set_postfix(avg_return=f"{float(avg_return):.3f}")
            total_return += float(avg_return)
            running_loss += loss_val
            steps += 1
            
        avg_return_epoch = total_return / steps
        loss_epoch = running_loss / steps
        print(f"Epoch {epoch+1}: loss {loss_epoch:.4f}, avg_return {avg_return_epoch:.3f}")
        agent.scheduler.step()

    # === Save Checkpoints ===
    checkpoint = {
        "model_state_dict": agent.policy.state_dict(),
        "market_params": {
            "alpha_weights": alpha_weights,
            "omega": omega,
            "return_weights": return_weights,
            "sigma": sigma,
            "trader_risk": trader_risk,
            "dealer_risk": dealer_risk,
            "horizon": horizon,
        },
    }
    torch.save(checkpoint, "checkpoint_multiasset.pt")
    
    # ==========================================
    # === SANITY CHECK EXPORT START ===
    # ==========================================
    print("\n--- Exporting Sanity Check Data ---")

    # 1. Export Environment Parameters to JSON
    # We export the exact matrices used in the env to verify against Analytical solution
    
    # Calculate Cost Lambda explicitly if it's not stored directly as a public attribute in all versions
    # Based on your previous code: cost_lambda = trader_risk * sigma (Assuming Assumption A)
    cost_lambda = env.cost_lambda if hasattr(env, 'cost_lambda') else (trader_risk * sigma)

    env_params = {
        "scalars": {
            "horizon": horizon,
            "trader_risk": trader_risk,
            "dealer_risk": dealer_risk,
            "num_assets": num_assets,
            "num_alphas": num_alphas
        },
        "matrices": {
            "alpha_weights": tensor_to_list(alpha_weights), # Matrix A (Mean reversion)
            "return_weights": tensor_to_list(return_weights), # Matrix B (Prediction)
            "sigma": tensor_to_list(sigma), # Covariance of Returns
            "omega": tensor_to_list(omega), # Covariance of Alphas
            "cost_lambda": tensor_to_list(cost_lambda) # Transaction Cost Matrix
        }
    }

    with open("debug_params.json", "w") as f:
        json.dump(env_params, f, indent=4)
    print("Saved 'debug_params.json'")

    # 2. Export a Single Trajectory to CSV
    # We run one episode without gradients to see what the agent actually does
    
    # Create a clean validation noise (Batch size = 1)
    U_test = torch.randn(1, horizon, num_alphas, device=device)
    state = env.reset(1)
    
    trajectory_data = []

    with torch.no_grad():
        for t in range(horizon):
            # Extract current state components
            # State structure: [alpha (num_alphas), last_position (num_assets)]
            current_alpha = state[0, :num_alphas].cpu().numpy()
            current_position = state[0, num_alphas:].cpu().numpy()
            
            # Get Action from Agent
            action = agent.policy(state)
            next_position = action[0].cpu().numpy() # This is w_t
            
            # Record Data
            row = {"t": t}
            # Add Alphas
            for i in range(num_alphas):
                row[f"alpha_{i}"] = current_alpha[i]
            # Add Current Holdings (before trading)
            for i in range(num_assets):
                row[f"pos_prev_{i}"] = current_position[i]
            # Add New Holdings (after trading)
            for i in range(num_assets):
                row[f"pos_new_{i}"] = next_position[i]
                
            trajectory_data.append(row)
            
            # Step Environment
            state = env.transition(state, action, U_test[:, t, :])

    # Save to CSV
    df = pd.DataFrame(trajectory_data)
    df.to_csv("debug_trajectory.csv", index=False)
    print("Saved 'debug_trajectory.csv'")
    print("==========================================")

if __name__ == "__main__":
    main()