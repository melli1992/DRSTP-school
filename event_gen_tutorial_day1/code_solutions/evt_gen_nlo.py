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
def mc_slicing(N, delta, xuniform):
  if(xuniform):
    # method 1 to get our x values (the easy way)
    x = [random.uniform(0, 1) for _ in range(N)]
    x = np.array(x)

    # only consider the region where x > delta
    # note that this introduces a jacobian!
    mask = x > delta
    x_large = x[mask]
    jac = (1-delta)

    # Integrand: (R(x)/x) for x > delta
    integrand_values = R(x_large) / x_large  
  else:
    # method 2 to get our x values (the better way)
    # we should sample x logarithmically because
    # we know that the integrand is ~dx/x
    # note that for subtraction one cannot simply do
    # this, because there we really need to go to x->0
    u = np.random.uniform(0, 1, N)

    # log-uniform samples in [delta, 1]
    # lnx = log(delta) - u * log(delta); |log(delta)| = jacobian
    # x = delta * e^u*log(1/delta) = delta * (1/delta)^u
    x   = delta * (1 / delta) ** u
    jac = np.log(1./delta) 

    # evaluate, note that sampling is now dlnx = dx/x
    # so remove the log(x)!
    integrand_values = R(x)

  integral_above = jac * np.mean(integrand_values)
  # this integral contains -B*log(delta) 
  
  # Small-x region (analytic)
  # ∫_0^δ dx x^(2eps) (B/x) = B(-1/2eps + ln δ + O(eps) + O(δ))
  delta_contrib     = B * np.log(delta)
  remainder_contrib = B/2*delta*(delta-2)
  integral_below    = delta_contrib + remainder_contrib
  print("Integral above slice ", integral_above, " exact answer = ", -B*np.log(delta) + B/2.*(-1+delta)**2)
  print("Integral below slice ", integral_below)

  # Combine 
  sigma = B + alpha * (
      Vf
      + integral_below
      + integral_above
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
    print("================")
    # Subtraction
    sigma_sub = mc_subtraction(N)
    print("Subtraction MC:", sigma_sub, " diff from exact:", sigma_sub - exact_res)
    print("================")
    # Slicing with different deltas (breaks down for too low values because of numerical precision)
    for delta in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6,  1e-7,  1e-8,  1e-10,  1e-11,  1e-12,  1e-13,  1e-14]:
        sigma_slice = mc_slicing(N, delta, True)
        print(f"Slicing MC (uniform in x, delta={delta}):", sigma_slice, " diff from exact:", sigma_slice - exact_res)
    print("================")
    for delta in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6,  1e-7,  1e-8,  1e-10,  1e-11,  1e-12,  1e-13,  1e-14]:
        sigma_slice = mc_slicing(N, delta, False)
        print(f"Slicing MC (uniform in lnx, delta={delta}):", sigma_slice, " diff from exact:", sigma_slice - exact_res)