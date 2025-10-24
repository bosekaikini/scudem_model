import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit # <--- NEW IMPORT

# ======================================================================
# 1. THEORETICAL FUNCTION FOR FITTING
# ======================================================================

def fit_function(t, A, B, C):
    """
    Generalized algebraic model for Gaussian collapse:
    sigma(t) = A * sqrt(1 / (1 + t/B)) + C
    
    A: Scaling factor (relates to sigma_0)
    B: Time scale factor (relates to n)
    C: Residual floor/offset
    """
    # Guard against division by zero if t/B = -1, though unlikely with our setup
    return A * np.sqrt(1 / (1 + t / B)) + C

# ======================================================================
# 2. HELPER FUNCTIONS AND PARAMETERS
# ======================================================================

# Initial parameters
mu_0 = 0
sigma_0 = 1 # Initial standard deviation

def generate_normal_samples(mu, sigma, n):
    rng = np.random.default_rng() 
    return rng.normal(mu, sigma, n)

def ml_estimate(samples):
    return np.mean(samples), np.std(samples, ddof=1)

n = 100  # Number of samples per generation (M)
num_trajectories = 25   
num_generations = 2500 

means, stds = [], []
pbar = tqdm(total=num_trajectories * num_generations)

# ======================================================================
# 3. RUN SIMULATION AND COLLECT DATA
# ======================================================================

# Run simulation (unchanged)
for _ in range(num_trajectories):
    mu, sigma = mu_0, sigma_0
    mean_arr, std_arr = [], []

    for _ in range(num_generations):
        samples = generate_normal_samples(mu, sigma, n)
        mu, sigma = ml_estimate(samples)
        mean_arr.append(mu)
        std_arr.append(sigma)
        pbar.update(1)

    means.append(mean_arr)
    stds.append(std_arr)

pbar.close()

std_data_array = np.array(stds)
ensemble_average_stds = np.mean(std_data_array, axis=0)

# ======================================================================
# 4. PERFORM OPTIMIZATION (CURVE FITTING)
# ======================================================================

generations = np.arange(num_generations) # Our independent variable (t)
data_to_fit = ensemble_average_stds # Our dependent variable (sigma_t)

# Define initial guesses for the parameters (A, B, C)
# A should be close to sigma_0=1.0
# B should be close to n=100
# C should be close to the residual std (0)
p0 = [sigma_0, n, 0.0]

# Perform non-linear least squares fit
# popt: optimal parameter values (A, B, C)
# pcov: covariance matrix of the parameters
try:
    popt, pcov = curve_fit(fit_function, generations, data_to_fit, p0=p0, maxfev=5000)
    A_opt, B_opt, C_opt = popt
    
    print("\n--- Optimized Model Parameters ---")
    print(f"A (Scaling, should be ~1.0): {A_opt:.4f}")
    print(f"B (Time Scale, should be ~n): {B_opt:.4f}")
    print(f"C (Residual Floor): {C_opt:.6f}")
    
except RuntimeError:
    print("\nError: Optimal parameters not found. Check initial guess or data.")
    A_opt, B_opt, C_opt = sigma_0, n, 0.0 # Use default if fitting fails

# ======================================================================
# 5. PLOT BOTH LINES
# ======================================================================

# Generate the fitted theoretical data using the optimized parameters
fitted_stds = fit_function(generations, A_opt, B_opt, C_opt)

# --- Plotting the Comparison ---

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot 1: Ensemble Average (Simulation Result)
ax.plot(
    generations,
    ensemble_average_stds, 
    color='blue', 
    label=f'Simulation Average ({num_trajectories} Trajectories)',
    linewidth=2
)

# Plot 2: Optimized Theoretical Function
ax.plot(
    generations,
    fitted_stds, 
    color='red', 
    linestyle='--', 
    label=f'Optimized Theoretical Fit: $\\sigma(t) = {A_opt:.3f}\\sqrt{{1/ (1+t/{B_opt:.1f})}} + {C_opt:.4f}$',
    linewidth=2
)

ax.set_xlabel('Generation ($t$)')
ax.set_ylabel(r'Standard Deviation ($\sigma_t$)')
ax.set_title('Gaussian Model Collapse: Simulation vs. Optimized Algebraic Fit')
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('normal_collapse_optimized_fit_plot.png')
plt.show()