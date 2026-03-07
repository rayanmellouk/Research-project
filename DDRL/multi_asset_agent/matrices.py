# === Multi-asset environment parameter utilities (Section 2.6) ===

import torch


def zeta_dim(num_assets, num_alphas):
    """Dimension of the flattened environment parameter vector ζ.
    Layout: [A_flat(K²), B_flat(S*K), Sigma_flat(S²), L_omega_flat(K²), trader_risk(1), cost_lambda_flat(S²)]
    """
    S, K = num_assets, num_alphas
    return 2 * K * K + S * K + 2 * S * S + 1


def flatten_env_params(A, B, Sigma, Omega, trader_risk):
    """Flatten raw environment matrices into a single ζ vector.

    Precomputes L_omega (Cholesky of Omega) and cost_lambda (trader_risk * Sigma).
    """
    device, dtype = A.device, A.dtype
    L_omega = torch.linalg.cholesky(Omega)
    cost_lambda = trader_risk * Sigma
    if torch.is_tensor(trader_risk):
        tr = trader_risk.view(1).to(device=device, dtype=dtype)
    else:
        tr = torch.tensor([trader_risk], device=device, dtype=dtype)
    return torch.cat([
        A.flatten(),
        B.flatten(),
        Sigma.flatten(),
        L_omega.flatten(),
        tr,
        cost_lambda.flatten(),
    ])


def unflatten_env_params_batch(zeta_batch, num_assets, num_alphas):
    """Reconstruct batched env params from zeta_batch of shape (B, D_zeta)."""
    S, K = num_assets, num_alphas
    idx = 0
    A = zeta_batch[:, idx:idx + K * K].reshape(-1, K, K); idx += K * K
    B = zeta_batch[:, idx:idx + S * K].reshape(-1, S, K); idx += S * K
    Sigma = zeta_batch[:, idx:idx + S * S].reshape(-1, S, S); idx += S * S
    L_omega = zeta_batch[:, idx:idx + K * K].reshape(-1, K, K); idx += K * K
    trader_risk = zeta_batch[:, idx]; idx += 1
    cost_lambda = zeta_batch[:, idx:idx + S * S].reshape(-1, S, S); idx += S * S
    return A, B, Sigma, L_omega, trader_risk, cost_lambda


def batched_random_correlation(N, n, device, dtype, rank, eps=1e-3):
    """Produce N random PSD correlation matrices of size (n, n)."""
    r = min(rank, n)
    X = torch.randn(N, n, r, device=device, dtype=dtype)
    C = X @ X.transpose(1, 2)  # (N, n, n)
    d = torch.sqrt(torch.diagonal(C, dim1=1, dim2=2).clamp_min(1e-12))  # (N, n)
    C = C / (d.unsqueeze(2) * d.unsqueeze(1))
    C = 0.5 * (C + C.transpose(1, 2))
    C = C + eps * torch.eye(n, device=device, dtype=dtype).unsqueeze(0)
    d = torch.sqrt(torch.diagonal(C, dim1=1, dim2=2).clamp_min(1e-12))
    C = C / (d.unsqueeze(2) * d.unsqueeze(1))
    return C


def sample_env_params_batch(batch_size, num_assets, num_alphas, device,
                             half_life_range=(2.0, 30.0),
                             asset_vol_range=(0.01, 0.04),
                             factor_vol_range=(0.5, 1.5),
                             pred_std_frac=0.1,
                             b_init_scale=0.1,
                             trader_risk_range=(0.1, 1.0),
                             sigma_corr_rank=None,
                             omega_corr_rank=None):
    """Sample a batch of random environment parameters as flat ζ vectors.

    Fully vectorized: uses batched correlation generation and the analytical
    stationary-variance solution (discrete Lyapunov for diagonal A) instead
    of a per-sample Monte Carlo simulation.

    Returns zeta_batch of shape (batch_size, D_zeta).
    """
    S, K = num_assets, num_alphas
    dtype = torch.float32
    N = batch_size
    sigma_rank = sigma_corr_rank if sigma_corr_rank is not None else min(3, S)
    omega_rank = omega_corr_rank if omega_corr_rank is not None else min(5, K)

    # --- A: batched diagonal persistence ---
    half_lives = half_life_range[0] + (half_life_range[1] - half_life_range[0]) * torch.rand(N, K, device=device, dtype=dtype)
    rhos = 2.0 ** (-1.0 / half_lives)  # (N, K)
    A = torch.diag_embed(rhos)  # (N, K, K)

    # --- Omega: batched correlation + vol ---
    fvol = factor_vol_range[0] + (factor_vol_range[1] - factor_vol_range[0]) * torch.rand(N, K, device=device, dtype=dtype)
    C_omega = batched_random_correlation(N, K, device, dtype, rank=omega_rank)
    D_f = torch.diag_embed(fvol)  # (N, K, K)
    Omega = D_f @ C_omega @ D_f
    Omega = 0.5 * (Omega + Omega.transpose(1, 2))
    Omega = Omega + 1e-8 * torch.eye(K, device=device, dtype=dtype).unsqueeze(0)

    # --- Sigma: batched correlation + vol ---
    avol = asset_vol_range[0] + (asset_vol_range[1] - asset_vol_range[0]) * torch.rand(N, S, device=device, dtype=dtype)
    C_sigma = batched_random_correlation(N, S, device, dtype, rank=sigma_rank)
    D_a = torch.diag_embed(avol)  # (N, S, S)
    Sigma = D_a @ C_sigma @ D_a
    Sigma = 0.5 * (Sigma + Sigma.transpose(1, 2))
    Sigma = Sigma + 1e-8 * torch.eye(S, device=device, dtype=dtype).unsqueeze(0)

    # --- B: batched scaling using analytical Lyapunov (diagonal A) ---
    # Stationary covariance: Sigma_f[i,j] = Omega[i,j] / (1 - rho_i * rho_j)
    B = b_init_scale * torch.randn(N, S, K, device=device, dtype=dtype)
    target_pred_std = avol.mean(dim=1) * pred_std_frac  # (N,)
    denom = 1.0 - rhos.unsqueeze(2) * rhos.unsqueeze(1)  # (N, K, K)
    Sigma_f = Omega / denom  # (N, K, K)
    pred_var = B @ Sigma_f @ B.transpose(1, 2)  # (N, S, S)
    pred_std = torch.sqrt(torch.diagonal(pred_var, dim1=1, dim2=2).clamp_min(1e-12)).mean(dim=1)  # (N,)
    scale = (target_pred_std / pred_std).unsqueeze(1).unsqueeze(2)  # (N, 1, 1)
    B = B * scale

    # --- trader_risk ---
    trader_risk = trader_risk_range[0] + (trader_risk_range[1] - trader_risk_range[0]) * torch.rand(N, device=device, dtype=dtype)

    # --- Flatten: batched Cholesky + concat ---
    L_omega = torch.linalg.cholesky(Omega)  # (N, K, K)
    cost_lambda = trader_risk.view(N, 1, 1) * Sigma  # (N, S, S)

    zeta = torch.cat([
        A.reshape(N, -1),
        B.reshape(N, -1),
        Sigma.reshape(N, -1),
        L_omega.reshape(N, -1),
        trader_risk.unsqueeze(1),
        cost_lambda.reshape(N, -1),
    ], dim=1)

    return zeta


# ------------------ Reproducible environment I/O ------------------


def save_env_params(path, A, B, Sigma, Omega, trader_risk,
                    num_assets, num_alphas, horizon, extra=None):
    """Save environment parameters to JSON for reproducible evaluation."""
    import json
    tr = float(trader_risk.item()) if torch.is_tensor(trader_risk) else float(trader_risk)
    params = {
        "scalars": {
            "horizon": int(horizon),
            "trader_risk": tr,
            "num_assets": int(num_assets),
            "num_alphas": int(num_alphas),
        },
        "matrices": {
            "A": A.detach().cpu().tolist(),
            "B": B.detach().cpu().tolist(),
            "Sigma": Sigma.detach().cpu().tolist(),
            "Omega": Omega.detach().cpu().tolist(),
        },
    }
    if extra:
        params["extra"] = extra
    with open(path, "w") as f:
        json.dump(params, f, indent=4)


def load_env_params(path, device="cpu"):
    """Load environment parameters from JSON and return (A, B, Sigma, Omega, trader_risk)."""
    import json
    with open(path) as f:
        params = json.load(f)
    dtype = torch.float32
    A = torch.tensor(params["matrices"]["A"], device=device, dtype=dtype)
    B = torch.tensor(params["matrices"]["B"], device=device, dtype=dtype)
    Sigma = torch.tensor(params["matrices"]["Sigma"], device=device, dtype=dtype)
    Omega = torch.tensor(params["matrices"]["Omega"], device=device, dtype=dtype)
    trader_risk = params["scalars"]["trader_risk"]
    return A, B, Sigma, Omega, trader_risk


def zeta_from_json(path, device="cpu"):
    """Load environment parameters from JSON and return a zeta vector of shape (1, D_zeta)."""
    A, B, Sigma, Omega, trader_risk = load_env_params(path, device=device)
    return flatten_env_params(A, B, Sigma, Omega, trader_risk).unsqueeze(0)
