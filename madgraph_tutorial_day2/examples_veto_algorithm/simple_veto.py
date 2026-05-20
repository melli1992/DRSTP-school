import numpy as np
import matplotlib.pyplot as plt

# Number of samples
N = 100000

# Uniform random numbers in (0,1)
r = np.random.rand(N)

# Sample from p(t) = 2 t exp(-t^2)
t = np.sqrt(-np.log(r))

# Plot histogram
bins = np.linspace(0, 4, 100)

plt.hist(t, bins=bins, density=True, alpha=0.6, label='Monte Carlo')

##############################
# Exact distribution
x = np.linspace(0, 4, 400)
p = 2 * x * np.exp(-x**2)

plt.plot(x, p, linewidth=2, label=r'$p(t) = 2 t e^{-t^2}$')

plt.xlabel('t')
plt.ylabel('Probability density')
plt.legend()
plt.show()