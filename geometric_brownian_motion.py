
import numpy as np
import matplotlib.pyplot as plt

def simulate_geometric_brownian_motion(S0, T=2, N=1000, mu=0.1, sigma=0.05):

  dt = T/N
  t = np.linspace(0, T, N)

  # standard norm N(0,1)
  W = np.random.standard_normal(size = N)

  # If 𝑍∼𝑁(0,1) Z∼N(0,1), then 𝑎𝑍∼𝑁(0,𝑎2)aZ∼N(0,a 2) therefore
  # N(0,dt) = sqrt(dt) * N(0,1)
  # solving for dWt, not just Wt so its N(0,dt) instead of N(0,t)

  W = np.cumsum(W) * np.sqrt(dt)
  X = (mu - 0.5 * sigma ** 2) * t + sigma * W
  S = S0 * np.exp(X)

  return t, S


def plot_simulation(t, S):
  plt.plot(t, S)
  plt.xlabel("Time (t)")
  plt.ylabel("Stock Price S(t)")
  plt.title("Geometric Brownian Motion")
  plt.show()


if __name__ == "__main__":
  
  time, data = simulate_geometric_brownian_motion(10)
  plot_simulation(time, data)