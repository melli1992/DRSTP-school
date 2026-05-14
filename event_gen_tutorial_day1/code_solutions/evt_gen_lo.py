import random
import numpy as np
import matplotlib.pyplot as plt

import time


random.seed(10)

def f(x):
    return 1 + x**2

def soln1(nsamples=100000):
  def sample_x():
    # the normalisation constant is 8./3.
    # so the normalised PDF is 3/8*(1+x^2)
    r = random.uniform(0, 1)

    # solve int_xmin^x dx f(x) = r int_xmin^xmax dx f(x)
    # gives us x^3 + 3x + (4 - 8r) = 0 == g(x)
    def g(x):
        return x**3 + 3*x + (4 - 8*r)

    # Bisection in [-1, 1]
    a, b = -1.0, 1.0
    for _ in range(50):  # enough iterations for precision
        m = 0.5 * (a + b)
        if g(m) > 0:
            b = m
        else:
            a = m

    return 0.5 * (a + b)
  
  # Generate samples
  samples = [sample_x() for _ in range(nsamples)]
  return samples

def soln2(nsamples=100000):
  fmax = 2
  def sample_x():
    while True:
      # Step 1: propose x uniformly
      x = random.uniform(-1, 1)
      
      # Step 2: uniform random for acceptance
      r = random.uniform(0, 1)
      
      # Step 3: acceptance condition
      if r < f(x) / fmax:
          return x

  # Generate many samples
  samples = [sample_x() for _ in range(nsamples)]
  return samples

# compute integral of f(x) from -1 to 1 numerically
def soln3(nsamples):
  sumf = 0
  def sample_x():
    return random.uniform(-1, 1)
  xs = [sample_x() for _ in range(nsamples)]
  fx = [f(x) for x in xs]
  return sum(fx)/nsamples*2 # *2 because we sampled between 0,1 so this is a jacobian!
   
# compute integral of f(x) from -1 to 1 numerically

# sum of f(x)
sumf    = 0
# total amount of trials
ntrials = 0
def soln4(nsamples):
  fmax = 2
  def sample_x():
    global sumf, ntrials
    while True:
      ntrials += 1
      # Step 1: propose x uniformly
      x = random.uniform(-1, 1)
      
      # Step 2: uniform random for acceptance
      r = random.uniform(0, 1)
      
      # Step 3: acceptance condition
      if r < f(x) / fmax:
        sumf += 2*fmax #2 is the jacobian for sampling between -1 and 1
        return x
  xs = [sample_x() for _ in range(nsamples)]
  print(f"Acceptance rate: {nsamples/ntrials:.4f}")
  integral = sumf/ntrials
  print(f"Integral is: {sumf/ntrials:.4f}")
  return xs, integral 

# --- timing wrapper ---
def time_sampler(n_samples, sampler):
  start = time.perf_counter()

  data = [sampler() for _ in range(n_samples)]

  end = time.perf_counter()

  elapsed = end - start
  print(f"Time: {elapsed:.4f} sec")
  print(f"Samples/sec: {n_samples / elapsed:.2f}")

  return data

nsamples = 100000
samples1 = time_sampler(nsamples, lambda: soln1(1)[0])
samples2 = time_sampler(nsamples, lambda: soln2(1)[0])
print("Integral:", soln3(nsamples))
samples3, integral = soln4(nsamples)
# ---- Plot histogram ----
plt.hist(samples1, bins=50, density=True, alpha=0.6, label="solution 1")
# plt.hist(samples2, bins=50, density=True, alpha=0.6, label="solution 2")
plt.hist(samples3, bins=50, density=True, alpha=0.6, label="solution 3")

# ---- Plot true distribution ----
x = np.linspace(-1, 1, 400)
pdf = (3/8) * f(x)

plt.plot(x, pdf, linewidth=2, label="True PDF")

# ---- Labels ----
plt.xlabel("x")
plt.ylabel("Probability density")
plt.title("Sampling from p(x) = (3/8)(1 + x^2)")
plt.legend()

plt.show()


# now last one
def soln5(nsamples):
  samples3, integral = soln4(nsamples)
  aEM = 1./137
  nc = 3.
  Qu = 2./3
  Qd = -1./3
  charge_sum = nc*(2*Qu**2 + 3*Qd**2)
  GeV2_to_pb = 0.389379*1e9
  prefact = 4*np.pi*charge_sum*aEM**2/3*GeV2_to_pb
  sqrt_ss = np.linspace(1,100,100)

  # now plot sigma(s) as a function of sqrt(s)
  sigma = prefact/sqrt_ss**2 * integral
  plt.plot(sqrt_ss, sigma)
  plt.xlabel(r"$\sqrt{s}$ [GeV]")
  plt.ylabel(r"$\sigma$ [pb]")
  plt.title("Cross section for $e^+e^- \\to$ hadrons")
  plt.xscale("log")
  plt.yscale("log")
  plt.show()

soln5(nsamples)
  
