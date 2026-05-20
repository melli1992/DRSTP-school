import numpy as np
import matplotlib.pyplot as plt

# Lower cutoff
t0 = 1.0

# Number of samples
N = 100000

# Uniform random numbers
r = np.random.rand(N)

# Sample from p(t) = 1/t Exp[-Int[1/t',{t',t0,t}]] = t0 / t^2
t = t0 / r

# Histogram
bins = np.logspace(0, 3, 100)

plt.hist(t, bins=bins, density=True,
         alpha=0.6, label='Monte Carlo')

##############################
# Exact distribution
x = np.logspace(0, 3, 400)
p = t0 / x**2

plt.plot(x, p, linewidth=2,
         label=r'$p(t) = t_0/t^2$')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('t')
plt.ylabel('Probability density')
plt.legend()
plt.show()