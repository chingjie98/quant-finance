import numpy as np
import matplotlib.pyplot as plt

"""
  initial_rate: Starting interest rate (r0)
  mean_reversion_speed: kappa: how quickly rates revert to the long-term mean
  long_term_mean: theta: the long-term average interest rate
  volatility: sigma: the randomness in rate movements
  total_time: total time horizon in years
"""

def simulate_vasicek_path(initial_rate, mean_reversion_speed, long_term_mean, volatility, total_time=1.0, steps=1000):
  dt = total_time / steps                      
  time_points = np.linspace(0, total_time, steps + 1)
  interest_rates = [initial_rate]             

  for _ in range(steps):
      current_rate = interest_rates[-1]
      drift = mean_reversion_speed * (long_term_mean - current_rate) * dt
      shock = volatility * np.sqrt(dt) * np.random.normal()
      next_rate = current_rate + drift + shock
      interest_rates.append(next_rate)

  return time_points, interest_rates


def plot_interest_rate_path(time_points, interest_rates):
    plt.plot(time_points, interest_rates)
    plt.xlabel('Time (Years)')
    plt.ylabel('Interest Rate')
    plt.title('Simulated Interest Rate Path (Vasicek Model)')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    time, rate_path = simulate_vasicek_path(
        initial_rate=1.3,
        mean_reversion_speed=0.9,
        long_term_mean=1.4,
        volatility=0.05
    )

    plot_interest_rate_path(time, rate_path)