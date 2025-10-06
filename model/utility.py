import numpy as np
from model.midprice import MidPriceModel

class Utility:
    ################################################################################
    # Since the simulation only requires the frozen reservation and no utility use it is only implemented for the sake of analysis.
    # TODO: Implement symmetric and inventory based utility functions.
    # The utility class can be used for future extensions of the project where we dont have an explicit expression for the reservation price but need to solve the PDE numerically.
     
    def __init__(self,strategy:str ,with_trading:bool=True, midprice:MidPriceModel=None, **kwargs):

        self.strategy = strategy
        self.param_check(strategy,kwargs)
        self.midprice_model = midprice if midprice is not None else MidPriceModel(
            initial_price=kwargs.get('initial_price',100.0),
            volatility=kwargs.get('volatility',2),
            time_step=kwargs.get('time_step',0.005),
            time_horizon=kwargs.get('time_horizon',1.0)
        )
        self.with_trading = with_trading

        self.wealth = kwargs.get('wealth',1000)
        self.risk_aversion = kwargs.get('risk_aversion',0.1)
        self.time_horizon = self.midprice_model.T
        self.time_step = self.midprice_model.dt
        self.volatility = self.midprice_model.sigma
        self.inventory = kwargs.get('inventory',0)


        
    def param_check(self,strategy,params):
        match strategy:
            case "frozen":
                required_params = ['wealth', 'risk_aversion', 'inventory']
                for param in required_params:
                    if param not in params:
                        raise ValueError(f"Missing required parameter '{param}' for strategy '{strategy}'")
            case "inventory":
                required_params = ['wealth', 'risk_aversion', 'inventory',"price_impact_model"]
                for param in required_params:
                    if param not in params:
                        raise ValueError(f"Missing required parameter '{param}' for strategy '{strategy}'")
            case "symmetric":
                pass #TODO: Implement symmetric strategy parameter check
            case _:
                raise ValueError(f"Invalid strategy '{strategy}'. Supported strategies are 'frozen', 'inventory', and 'symmetric'")

    def compute_frozen_utility(self):
        q = self.inventory
        sigma = self.midprice_model.sigma
        gamma = self.risk_aversion
        x = self.wealth
        t = self.midprice_model.current_time
        s = self.midprice_model.current_price
        return -np.exp(-gamma(x+q*s-0.5*gamma*(sigma**2)*(self.time_horizon-t)*(q**2)))
    
    def compute_frozen_ask(self):
        q = self.inventory
        sigma = self.midprice_model.sigma
        gamma = self.risk_aversion
        x = self.wealth
        t = self.midprice_model.current_time
        s = self.midprice_model.current_price
        return s + (1-2*q)*gamma*(sigma**2)*(self.time_horizon-t)
    
    def compute_frozen_bid(self):
        q = self.inventory
        sigma = self.midprice_model.sigma
        gamma = self.risk_aversion
        x = self.wealth
        t = self.midprice_model.current_time
        s = self.midprice_model.current_price
        return s + (-1-2*q)*gamma*(sigma**2)*(self.time_horizon-t)
    
    def compute_frozen_indifference_price(self):
        return (self.compute_frozen_bid() + self.compute_frozen_ask())*0.5
    
