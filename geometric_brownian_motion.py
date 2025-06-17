
import numpy as np

def simulate_geometric_brownian_motion(S0, T=2, N=1000, mu=0.1, sigma=0.05):

  dt = T/N
  t = np.linspace(0, T, N)

  # standard norm N(0,1)
  W = np.random.standard_normal(size = N)

  # N(0,dt) = sqrt(dt) * N(0,1)
  W = np.cumsum(W) * np.sqrt(dt)
  X = (mu - 0.5 * sigma ** 2) * t + sigma * W
  S = S0 * np.exp(X)

  return t, S


