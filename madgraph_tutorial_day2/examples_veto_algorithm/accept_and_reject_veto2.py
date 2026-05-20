import numpy as np
import matplotlib.pyplot as plt


# Lower cutoff
t0 = 1.0

# the function
def f(t):
    return 1/t

# the overestimate
def g(t,t0):
    return 1/t0

# the veto algorithm
def veto_algo(t0=1.0):
    # 1. start at t = t0
    t = t0
    while True:
        # 2. Select new ti
        r1 = np.random.rand()
        dt = -t0*np.log(r1)
        t += dt

        # 3. Compute f(ti)/g(ti) and compare with r2
        r2 = np.random.rand()
        if r2 < f(t)/g(t,t0):
            return t
        # else go back to 2. 


# Generate samples
N = 100000
samples = np.array([veto_algo(t0) for _ in range(N)])

# Histogram
bins = np.logspace(0, 3, 100)


plt.hist(samples, bins=bins, density=True,
         alpha=0.6, label='Monte Carlo')

##########################################
# Exact distribution
x = np.logspace(0, 3, 400)
p = t0 / x**2

plt.plot(x, p, linewidth=2,
         label=r'$t_0/t^2$')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('t')
plt.ylabel('Probability density')
plt.legend()
plt.show()