
import numpy as np
from math import *
from scipy import stats
import time

class OptionPricing:
  
  """
  S0: Spot price
  E: Exercise price
  T = Time to maturity (in years) 
  rf = risk-free rate
  sigma = volatility
  """

  def __init__(self,S0,E,T,rf,sigma):
    self.S0 = S0            
    self.E = E              
    self.T = T             
    self.rf = rf           
    self.sigma = sigma    
  
  def call_price(self):
    # d1 accounts for time value and volatility of the underlying asset
    d1 = (log(self.S0 / self.E) + (self.rf + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T))
    # d2 adjusts d1 downward by the volatility term
    d2 = d1 - self.sigma * sqrt(self.T)
    print(f"[Call] d1: {d1}, d2: {d2}")
    # Black-Scholes call option formula:
    # Call = S0 * N(d1) - E * e^(-rT) * N(d2)
    return self.S0 * stats.norm.cdf(d1) - self.E * exp(-self.rf * self.T) * stats.norm.cdf(d2)

  def put_price(self):
    # d1 accounts for time value and volatility of the underlying asset
    d1 = (log(self.S0 / self.E) + (self.rf + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T))
    # d2 adjusts d1 downward by the volatility term
    d2 = d1 - self.sigma * sqrt(self.T)
    print(f"[Put] d1: {d1}, d2: {d2}")
    # Black-Scholes put option formula:
    # Put = -S0 * N(-d1) + E * e^(-rT) * N(-d2)
    return -self.S0 * stats.norm.cdf(-d1) + self.E * exp(-self.rf * self.T) * stats.norm.cdf(-d2)
  
  def monte_carlo_call(self, num_simulations=10000):
    # Simulate terminal stock prices under risk-neutral GBM
    Z = np.random.standard_normal(num_simulations)
    ST = self.S0 * np.exp((self.rf - 0.5 * self.sigma**2) * self.T + self.sigma * sqrt(self.T) * Z)

    # Calculate discounted payoff for a call
    payoff = np.maximum(ST - self.E, 0)
    price = exp(-self.rf * self.T) * np.mean(payoff)
    print(f"[Call - Monte Carlo] Estimated Price: {price}")
    return price

  def monte_carlo_put(self, num_simulations=10000):
    # Simulate terminal stock prices under risk-neutral GBM
    Z = np.random.standard_normal(num_simulations)
    ST = self.S0 * np.exp((self.rf - 0.5 * self.sigma**2) * self.T + self.sigma * sqrt(self.T) * Z)

    # Calculate discounted payoff for a put
    payoff = np.maximum(self.E - ST, 0)
    price = exp(-self.rf * self.T) * np.mean(payoff)
    print(f"[Put - Monte Carlo] Estimated Price: {price}")
    return price


if __name__ == '__main__':
  option = OptionPricing(S0=100, E=100, T=1, rf=0.05, sigma=0.2)
  print("Call Option Price:", option.call_price())
  print("Put Option Price:", option.put_price())

  print("MC  Call Price:", option.monte_carlo_call())
  print("MC  Put Price :", option.monte_carlo_put())