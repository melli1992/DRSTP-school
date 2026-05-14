import numpy as np
import matplotlib.pyplot as plt


# Lower cutoff
t0 = 1.0

def f(t):
    return 1/t

def g(t,t0):
    return 1/t0
# helper function
def sample_veto_f_over_t(t0=1.0):
    t = t0

    while True:
        # 1. propose next event from g(t)=1
        u1 = np.random.rand()
        dt = -t0*np.log(u1)
        t += dt

        # 2. acceptance probability f/g = (1/t)/1
        u2 = np.random.rand()

        if u2 < f(t)/g(t,t0):
            return t


# Generate samples
N = 100000
samples = np.array([sample_veto_f_over_t(t0) for _ in range(N)])

# Histogram
bins = np.logspace(0, 3, 100)


plt.hist(samples, bins=bins, density=True,
         alpha=0.6, label='Monte Carlo')

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