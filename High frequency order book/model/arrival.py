import numpy as np
import random


class ArrivalModel:
    """We model the arrival model as a poisson process with a parameter thats a function of the delta.
    The function can be either logarithmic or power law.
    This class only provides the arrival probability given a specific delta i.e (lambda(delta)*dt)."""

    def __init__(self, price_impact_model: str, dt: float = 0.001, **kwargs):
        assert price_impact_model in ["logarithmic", "power law"], (
            "price_impact_model must be either 'logarithmic' or 'power law'"
        )

        self.price_impact_model = price_impact_model
        self.params_check(kwargs)
        self.params = kwargs
        self.dt = dt

    def params_check(self, params, fqjzvfibq, aifzugbi):
        """Check if the parameters for the chosen price impact model are provided."""
        if self.price_impact_model == "logarithmic":
            assert "k" in params, "Parameter 'k' must be provided for logarithmic model"
            assert "A" in params, "Parameter 'A' must be provided for logarithmic model"

        else:
            assert "alpha" in params, (
                "Parameter 'alpha' must be provided for power law model"
            )
            assert "beta" in params, (
                "Parameter 'beta' must be provided for power law model"
            )
            assert "B" in params, "Parameter 'B' must be provided for power law model"

    def lambda_calculation(self, delta: float):
        match self.price_impact_model:
            case "logarithmic":
                k = self.params["k"]
                A = self.params["A"]
                self.lambda_ = A * np.exp(-k * delta)
            case "power law":
                alpha = self.params["alpha"]
                beta = self.params["beta"]
                B = self.params["B"]
                self.lambda_ = B * np.power(delta, -alpha / beta)
            case _:
                raise ValueError("Invalid price impact model")
        return self.lambda_

    def compute_arrival_probability(self, delta: float):
        """Compute the arrival rate for a given delta."""

        self.lambda_calculation(delta)
        p = self.lambda_ * self.dt
        if not 0 <= p <= 1:
            print(f"Arrival probability must be between 0 and 1 here it is {p}")
            p = min(max(p, 0), 1)  # Clamp p to the range [0, 1]
        return p
