import random
import numpy as np 
from model.midprice import MidPriceModel
from model.arrival import ArrivalModel
from model.utility import Utility
# TODO: check for the non logarithmic price impact model if the formulas for the spread and reservation price are still valid. If not implement the symmetric strategy and change the formulas for the spread and reservation price in this class.
# TODO: is the with_trading parameter needed in the utility class?
class Strategy:
    """
    class to implement the full model's logic
    For now we only implement the model described in the paper i.e with the same approximations which give us the explicit formula for the reservation price which is always equal to the frozen reservation price and the spread which is the same for both the symmetric and inventory strategy."""

    def __init__(self,**kwargs):
        self.midprice_model = MidPriceModel(
            initial_price=kwargs.get('initial_price',100.0),
            volatility=kwargs.get('volatility',2),
            time_step=kwargs.get('time_step',0.005),
            time_horizon=kwargs.get('time_horizon',1.0)
        )

        self.utility_model = Utility(
            strategy=kwargs.get('strategy','inventory'),
            with_trading=kwargs.get('with_trading',True),
            midprice=self.midprice_model,
            wealth=kwargs.get('wealth',1000),
            risk_aversion=kwargs.get('risk_aversion',0.1),
            inventory=kwargs.get('inventory',0),
            price_impact_model=kwargs.get('price_impact_model','logarithmic')
        )
        
        self.arrival_model = ArrivalModel(
            dt=self.midprice_model.dt,
            **kwargs
        )

        self.time_horizon = self.midprice_model.T
        self.time_step = self.midprice_model.dt
        self.current_time = 0.0
        self.number_of_steps = self.midprice_model.number_of_steps

        self.inventory = self.utility_model.inventory
        self.wealth = self.utility_model.wealth
        self.risk_aversion = self.utility_model.risk_aversion
        self.volatility = self.midprice_model.sigma
        self.strategy = kwargs.get('strategy','frozen')
        self.with_trading = kwargs.get('with_trading',True)
        self.price_impact_model = kwargs.get('price_impact_model','logarithmic')

        self.trade_history = {"ask":[], "bid":[],"midprice":[]}
        self.inventory_history = []
        
        self.kwargs = kwargs    

    def compute_spread(self):
        """ Since in the paper the spread is the same for both the symmetric and inventory strategy, we calculate it here."""
        # Implement the formula obtained from a power law impact model??

        gamma = self.risk_aversion
        sigma = self.volatility
        k = self.arrival_model.params.get('k',1.5)  # default value of k is 1.5 if not provided
        t = self.current_time
        T = self.time_horizon
        spread = gamma * sigma**2 * (T - t) + (2/gamma)*np.log(1 + (gamma/k))
        return spread
    
    def compute_reservation_price(self):
        """ Since in the paper the reservation price is the same for both the symmetric and inventory strategy and since it is equal to the frozen reservation it is computable using the  utility class as it only depends on the state variables."""

        return self.utility_model.compute_frozen_indifference_price()
    
    def compute_optimal_quotes(self):
        """Compute the optimal bid and ask quotes based on the current strategy."""
        reservation_price = self.compute_reservation_price() if self.strategy == "inventory" else self.midprice_model.current_price
        spread = self.compute_spread()
        optimal_ask = reservation_price + spread / 2
        optimal_bid = reservation_price - spread / 2
        return optimal_bid, optimal_ask
    
    def step(self):
        """Advance the model by one time step."""
        if self.current_time >= self.time_horizon:
            print("Warning: current time is already at or past the time horizon. No further steps can be taken.")
            return
        optimal_bid, optimal_ask = self.compute_optimal_quotes()
        delta_a,delta_b = optimal_ask - self.midprice_model.current_price, self.midprice_model.current_price - optimal_bid
        p_a = self.arrival_model.compute_arrival_probability(delta_a)
        p_b = self.arrival_model.compute_arrival_probability(delta_b)
        if random.random() < p_a and self.with_trading:
            # Execute a sell order
            self.inventory -= 1
            self.utility_model.inventory = self.inventory
            self.wealth += optimal_ask
            self.utility_model.wealth = self.wealth
            self.trade_history["ask"].append((self.current_time, optimal_ask,"hit"))
        else:
            self.trade_history["ask"].append((self.current_time, optimal_ask,"no hit"))
        if random.random() < p_b and self.with_trading:
            # Execute a buy order
            self.inventory += 1
            self.utility_model.inventory = self.inventory
            self.wealth -= optimal_bid
            self.utility_model.wealth = self.wealth
            self.trade_history["bid"].append((self.current_time, optimal_bid,"lift"))
        else:
            self.trade_history["bid"].append((self.current_time, optimal_bid,"no lift"))
        self.inventory_history.append((self.current_time,self.inventory))
        self.trade_history["midprice"].append((self.current_time,self.midprice_model.current_price))
        self.midprice_model.step()
        self.current_time = self.midprice_model.current_time

    def reset(self):
        """Reset the model to its initial state."""
        self.midprice_model.reset()
        self.utility_model.inventory = self.kwargs.get('inventory',0)
        self.utility_model.wealth = self.kwargs.get('wealth',1000)
        self.inventory = self.utility_model.inventory
        self.wealth = self.utility_model.wealth
        self.current_time = 0.0
        self.trade_history = {"ask":[], "bid":[],"midprice":[]}
        self.inventory_history = []

    def run(self):
        """Run the model from the initial time until the time horizon is reached."""
        self.reset()
        while self.current_time < self.time_horizon:
            self.step()

    def plot_trade_history(self, show_fig=True, save_fig=False, save_path="trade_history.png",fig_name="Trade History"):
        assert len(self.trade_history["ask"]) > 0, "No trade history to plot. Please run the model first."

        import matplotlib.pyplot as plt

        ask_times, ask_prices, ask_status = zip(*self.trade_history["ask"])
        bid_times, bid_prices, bid_status = zip(*self.trade_history["bid"])
        midprice_times, midprices = zip(*self.trade_history["midprice"])

        # Compute reservation prices as simple average of bid and ask
        reservation_prices = [(a + b) / 2 for a, b in zip(ask_prices, bid_prices)]

        plt.figure(figsize=(12, 6))
        plt.plot(midprice_times, midprices, label='Midprice', color='black', linewidth=2, alpha=0.7)
        plt.plot(ask_times, reservation_prices, label='Reservation Price', color='green', linewidth=1.5, linestyle='--', alpha=0.8)
        plt.plot(ask_times, ask_prices, label='Ask Price', color='red', linewidth=1, alpha=0.6)
        plt.plot(bid_times, bid_prices, label='Bid Price', color='blue', linewidth=1, alpha=0.6)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.title(fig_name, fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_fig:
            plt.savefig(save_path)
        if show_fig:
            plt.show()
        plt.close()

