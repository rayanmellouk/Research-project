from model.strategy import Strategy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if __name__=="__main__":
    print("launching first run")
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
    strategy.plot_trade_history(save_fig=True,show_fig=True,save_path="tests/first_run.png",fig_name="First Run Trade History with gamma=0.1")

    print("launching the animated version of the first run")
    strategy.reset()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Animated Trade History with gamma=0.1")
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')

    midprice_line, = ax.plot([], [], label='Midprice', color='black')
    reservation_line, = ax.plot([], [], label='Reservation Price', color='green', linestyle='--')
    ask_line, = ax.plot([], [], label='Ask Price', color='red')
    bid_line, = ax.plot([], [], label='Bid Price', color='blue')

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    def init():
        midprice_line.set_data([], [])
        reservation_line.set_data([], [])
        ask_line.set_data([], [])
        bid_line.set_data([], [])
        return midprice_line, reservation_line, ask_line, bid_line

    def update(frame):
        if strategy.current_time < strategy.time_horizon:
            strategy.step()

        midprice_times, midprices = zip(*strategy.trade_history["midprice"]) if strategy.trade_history["midprice"] else ([], [])
        ask_times, ask_prices, _ = zip(*strategy.trade_history["ask"]) if strategy.trade_history["ask"] else ([], [], [])
        bid_times, bid_prices, _ = zip(*strategy.trade_history["bid"]) if strategy.trade_history["bid"] else ([], [], [])

        # Plot up to the current frame
        reservation_prices = [(a + b) / 2 for a, b in zip(ask_prices, bid_prices)] if ask_prices and bid_prices else []

        midprice_line.set_data(midprice_times, midprices)
        reservation_line.set_data(ask_times, reservation_prices)
        ask_line.set_data(ask_times, ask_prices)
        bid_line.set_data(bid_times, bid_prices)

        if midprice_times:
            ax.set_xlim(0, strategy.time_horizon)
        all_prices = list(midprices) + list(ask_prices) + list(bid_prices)
        if all_prices:
            ax.set_ylim(min(all_prices) - 1, max(all_prices) + 1)

        return midprice_line, reservation_line, ask_line, bid_line

    # Calculate frames to be one per time step
    n_frames = strategy.number_of_steps

    ani = FuncAnimation(
        fig, update, frames=n_frames, init_func=init, blit=True, interval=50, repeat=False
    )

    plt.show()
    plt.savefig("tests/first_run_animated.png")
