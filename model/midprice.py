import numpy as np 

class MidPriceModel():
    def __init__(self, initial_price=100.0, volatility=2,time_step=0.005,time_horizon=1.0):
        """Initialize the midprice model with given parameters."""
        

        self.current_price = initial_price
        self.sigma = volatility
        self.dt = time_step
        self.T = time_horizon
        self.s0 = initial_price
        self.initial_price = initial_price
        self.number_of_steps = int(self.T / self.dt)
        self.current_time = 0.0

        self.params_check()

    def params_check(self):
        """Check if the parameters are valid."""
        assert self.sigma > 0, "Volatility must be positive"
        assert self.dt > 0, "Time step must be positive"
        assert self.T > 0, "Time horizon must be positive"
        assert self.initial_price > 0, "Initial price must be positive"
        assert self.number_of_steps > 0, "Number of steps must be positive"

    def step(self):
        """Generate a single step in the midprice process."""
        dW = np.random.normal(0, np.sqrt(self.dt))   # note: We choose to increment the midprice with a normal variable with variance dt instead of an increment of +-sigma*sqrt(dt) with equal probability. This is because the former is a better approximation of the continuous process. Et en sah psq c'est deja sur numpy.
        ds = self.sigma * dW
        self.current_price += ds
        self.current_time += self.dt
        return self.current_price
    
    def value_at_time(self,t):
        """Get the midprice at a specific time t."""
        if self.current_time >= t:
            print(f"Warning: current time is already past the requested time. Returning the last explicitly known price i.e at time {self.current_time}.")
            return self.current_price
        while self.current_time < t:
            self.step()
        return self.current_price

    def reset(self):
        """Reset the model to its initial state."""
        self.current_price = self.initial_price
        self.current_time = 0.0
