# === Creates matrices return_weights (B), alpha_weights (Id - Phi or A) sigma, omega ===

import torch


def make_spd_from_vol_corr(vol: torch.Tensor, corr: torch.Tensor, eps: float = 1e-8):
    # vol: (n,), corr: (n,n) with diag=1 and PSD (or near-PSD)
    D = torch.diag(vol)
    cov = D @ corr @ D
    cov = 0.5 * (cov + cov.T)  # symmetrize
    cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    return cov


def random_correlation(
    n: int, device: str, dtype=torch.float32, rank: int = None, eps: float = 1e-3
):
    """
    Produce a PSD correlation-ish matrix.
    rank controls low-rank structure (None -> n).
    """
    r = n if rank is None else min(rank, n)
    X = torch.randn(n, r, device=device, dtype=dtype)
    C = X @ X.T
    # normalize to correlation
    d = torch.sqrt(torch.diag(C).clamp_min(1e-12))
    C = C / (d[:, None] * d[None, :])
    C = 0.5 * (C + C.T)
    # jitter for numerical stability
    C = C + eps * torch.eye(n, device=device, dtype=dtype)
    # renormalize diagonal to 1
    d = torch.sqrt(torch.diag(C).clamp_min(1e-12))
    C = C / (d[:, None] * d[None, :])
    return C


def make_A_diagonal(persistences: torch.Tensor):
    # persistences: (K,) with abs < 1
    return torch.diag(persistences)


def make_A_random_symmetric(
    K: int, max_abs_eig: float, device: str, dtype=torch.float32
):
    """
    Symmetric VAR(1) matrix A with eigenvalues in (-max_abs_eig, max_abs_eig),
    hence stable if max_abs_eig < 1.
    """
    # random orthogonal via QR
    M = torch.randn(K, K, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(M)
    eigs = (2 * torch.rand(K, device=device, dtype=dtype) - 1.0) * max_abs_eig
    A = Q @ torch.diag(eigs) @ Q.T
    A = 0.5 * (A + A.T)
    return A


def scale_B_to_target_pred_var(
    B: torch.Tensor, A: torch.Tensor, Omega: torch.Tensor, target_pred_std: float
):
    """
    Roughly scale B so that std of predictable component (B f_t) is about target_pred_std,
    averaged across assets. Uses Monte Carlo approximation for Var(f_t).
    """
    device, dtype = B.device, B.dtype
    K = A.shape[0]

    # simulate f under stationary dynamics for a bit
    T_burn, T = 200, 300
    f = torch.zeros(K, device=device, dtype=dtype)
    fs = []
    L = torch.linalg.cholesky(Omega)
    for t in range(T_burn + T):
        eps = L @ torch.randn(K, device=device, dtype=dtype)
        f = A @ f + eps
        if t >= T_burn:
            fs.append(f.clone())
    F = torch.stack(fs, dim=0)  # (T,K)
    pred = F @ B.T  # (T,S)
    pred_std = pred.std(dim=0).mean().clamp_min(1e-12)  # average across assets
    return B * (target_pred_std / pred_std)


# ------------------ Example general initialization ------------------


def make_market_params(num_assets: int, 
                       num_alphas: int, 
                       dealer_risk_range: tuple = (0.1, 1.0), 
                       trader_risk_range: tuple = (0.1, 1.0),
                       dealer_risk: float = None, 
                       trader_risk: float = None, 
                       device: str = "cpu"):
    dtype = torch.float32

    # --- A and Omega (predictor dynamics) ---
    # Multi-scale idea: persistences from slow -> fast
    # e.g. linearly spaced half-lives -> convert to rho
    half_lives = torch.linspace(
        30, 2, num_alphas, device=device, dtype=dtype
    )  # in steps; adjust as desired
    rhos = 2.0 ** (-1.0 / half_lives)  # rho = 2^(-1/h)
    A = make_A_diagonal(rhos)  # or make_A_random_symmetric(K, max_abs_eig=0.98, ...)

    # Factor shock correlation and vols
    factor_vol = torch.full(
        (num_alphas,), 1.0, device=device, dtype=dtype
    )  # set scale; will affect SNR
    C_omega = random_correlation(
        num_alphas, device=device, dtype=dtype, rank=min(5, num_alphas), eps=1e-3
    )
    Omega = make_spd_from_vol_corr(factor_vol, C_omega, eps=1e-8)

    # --- Sigma (return noise covariance) ---
    asset_vol = 0.02 * torch.ones(
        num_assets, device=device, dtype=dtype
    )  # per-step vol; calibrate to your dt
    C_sigma = random_correlation(
        num_assets, device=device, dtype=dtype, rank=min(3, num_assets), eps=1e-3
    )
    Sigma = make_spd_from_vol_corr(asset_vol, C_sigma, eps=1e-8)

    # --- B (return_weights) ---
    # start with random exposures, then scale to a target predictable-return magnitude
    B = 0.1 * torch.randn(num_assets, num_alphas, device=device, dtype=dtype)

    # choose target predictable component std per step (e.g., 10% of total return std)
    # total per-step std ~ mean(asset_vol); so predictable could be, say, 0.002 when vol is 0.02
    target_pred_std = float(asset_vol.mean().item()) * 0.1
    B = scale_B_to_target_pred_var(B, A, Omega, target_pred_std)

    if dealer_risk is None:
        lo = dealer_risk_range[0]
        hi = dealer_risk_range[1]
        dealer_risk = float((hi - lo) * torch.rand(1, device=device).item() + lo)
    if trader_risk is None:
        lo = trader_risk_range[0]
        hi = trader_risk_range[1]
        trader_risk = float((hi - lo) * torch.rand(1, device=device).item() + lo)

    return B, A, Sigma, Omega, trader_risk, dealer_risk


# ------------------ Variable environment helpers (Section 2.6) ------------------


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

    Mirrors make_market_params: half-life-based persistence, full correlation
    structure for Sigma and Omega, and B scaled to a target predictable-return
    standard deviation.

    Returns zeta_batch of shape (batch_size, D_zeta).
    """
    S, K = num_assets, num_alphas
    dtype = torch.float32
    sigma_rank = sigma_corr_rank if sigma_corr_rank is not None else min(3, S)
    omega_rank = omega_corr_rank if omega_corr_rank is not None else min(5, K)

    zetas = []
    for _ in range(batch_size):
        # --- A: half-life-based diagonal persistence ---
        half_lives = half_life_range[0] + (half_life_range[1] - half_life_range[0]) * torch.rand(K, device=device, dtype=dtype)
        rhos = 2.0 ** (-1.0 / half_lives)
        A = make_A_diagonal(rhos)

        # --- Omega: full correlation structure ---
        fvol = factor_vol_range[0] + (factor_vol_range[1] - factor_vol_range[0]) * torch.rand(K, device=device, dtype=dtype)
        C_omega = random_correlation(K, device=device, dtype=dtype, rank=omega_rank, eps=1e-3)
        Omega = make_spd_from_vol_corr(fvol, C_omega, eps=1e-8)

        # --- Sigma: full correlation structure ---
        avol = asset_vol_range[0] + (asset_vol_range[1] - asset_vol_range[0]) * torch.rand(S, device=device, dtype=dtype)
        C_sigma = random_correlation(S, device=device, dtype=dtype, rank=sigma_rank, eps=1e-3)
        Sigma = make_spd_from_vol_corr(avol, C_sigma, eps=1e-8)

        # --- B: random exposures scaled to target predictable-return std ---
        B = b_init_scale * torch.randn(S, K, device=device, dtype=dtype)
        target_pred_std = float(avol.mean().item()) * pred_std_frac
        B = scale_B_to_target_pred_var(B, A, Omega, target_pred_std)

        # --- trader_risk ---
        tr_val = float(trader_risk_range[0] + (trader_risk_range[1] - trader_risk_range[0]) * torch.rand(1, device=device).item())

        zetas.append(flatten_env_params(A, B, Sigma, Omega, tr_val))

    return torch.stack(zetas, dim=0)
