from model.strategy import Strategy


if __name__=="main":

    strategy = Strategy(
        initial_price=100.0,
        volatility=2,
        time_step=0.005,
        time_horizon=1.0,
        strategy='inventory',
        with_trading=True,
        wealth=1000,
        risk_aversion=0.1,
        inventory=0,
        price_impact_model='logarithmic',
        k=1.5,
        A=140
    )
    strategy.run() # Example for running a full trajectory with the exact same model as in the paper
