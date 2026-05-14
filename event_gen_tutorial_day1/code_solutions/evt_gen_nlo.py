import random
import numpy as np

# Parameters
B = 2.0
alpha = 0.1
Vf = 1.0

# Number of MC points
N = 100000
random.seed(10)
def R(x):
    return B * (1 + x * (1 - x))
# -------------------------
# Subtraction method
# -------------------------
def mc_subtraction(N):
  # Sample x uniformly in [0,1]
  x = [random.uniform(0, 1) for _ in range(N)]

  # Integrand: (R(x) - R(0)) / x = B*(1 - x)
  integrand_values = [(R(xi) - R(0))/xi for xi in x]

  # MC estimate of integral
  integral = np.mean(integrand_values)

  # Total cross section
  sigma = B + alpha * (Vf + integral)
  return sigma


# -------------------------
# Slicing method
# -------------------------
def mc_slicing(N, delta):
  # get our x values
  x = [random.uniform(0, 1) for _ in range(N)]
  x = np.array(x)
  # only consider the region where x > delta
  mask = x > delta
  x_large = x[mask]

  # Integrand: (R(x)/x) for x > delta
  integrand_values = [R(xi)/xi for xi in x_large]

  # MC estimate for large region
  integral = 0
  for val in integrand_values:
    integral += val
  integral_large = integral / len(x_large) if len(x_large) > 0 else 0
  
  # Small-x region (analytic)
  # ∫_0^δ dx x^(2eps) (B/x) = B(-1/2eps + ln δ + O(eps) + O(δ))
  slice_contrib = B * np.log(delta)
  exact_contrib = B*np.log(delta) + B/2*delta*(delta-2)

  # Combine 
  sigma = B + alpha * (
      Vf
      # + slice_contrib
      + exact_contrib
      + integral_large
  )

  return sigma


# -------------------------
# Exact result
# -------------------------
def exact():
    return B + alpha * (Vf + B / 2)


# -------------------------
# Run comparison
# -------------------------
if __name__ == "__main__":
    exact_res = exact()
    print("Exact result:", exact_res)

    # Subtraction
    sigma_sub = mc_subtraction(N)
    print("Subtraction MC:", sigma_sub, " diff from exact:", sigma_sub - exact_res)

    # Slicing with different deltas
    for delta in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        sigma_slice = mc_slicing(N, delta)
        print(f"Slicing MC (delta={delta}):", sigma_slice, " diff from exact:", sigma_slice - exact_res)