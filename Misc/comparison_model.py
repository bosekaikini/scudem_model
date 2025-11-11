import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ======================================================================
# 1. DEFINE YOUR THEORETICAL FUNCTION HERE
# ======================================================================

def theoretical_collapse_rate(t, n, sigma_0):
    return sigma_0 * np.sqrt(1 / (1 + t / n)) - 0.00008*t

# ======================================================================
# 2. HELPER FUNCTIONS AND PARAMETERS
# ======================================================================

# Initial parameters
mu_0 = 0
sigma_0 = 1 # Initial standard deviation (used by both simulation and theory)

def generate_normal_samples(mu, sigma, n):
    # Use default_rng for robust random number generation
    rng = np.random.default_rng() 
    return rng.normal(mu, sigma, n)

def ml_estimate(samples):
    # ddof=1 is used for unbiased standard deviation
    return np.mean(samples), np.std(samples, ddof=1)

n = 100  # Number of samples per generation (M in the paper)
num_trajectories = 25   
num_generations = 2500 

means, stds = [], []
pbar = tqdm(total=num_trajectories * num_generations)

# ======================================================================
# 3. RUN SIMULATION AND COLLECT DATA
# ======================================================================

# This section is INDEPENDENT of your theoretical_collapse_rate function.
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

# Convert the list of lists (stds) into a 2D NumPy array for calculation
std_data_array = np.array(stds)

# ======================================================================
# 4. CALCULATE AVERAGE AND PLOT BOTH LINES
# ======================================================================

# 4a. Calculate the Ensemble Average (Simulation Result)
ensemble_average_stds = np.mean(std_data_array, axis=0)

# 4b. Generate the data points for your theoretical function
generations = np.arange(num_generations)
# This calls your custom function from Section 1
theoretical_stds = theoretical_collapse_rate(generations, n, sigma_0)

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

# Plot 2: Theoretical Function (Your Model)
ax.plot(
    generations,
    theoretical_stds, 
    color='red', 
    linestyle='--', 
    label='Theoretical Collapse Rate (Your Model)',
    linewidth=2
)

ax.set_xlabel('Generation ($t$)')
ax.set_ylabel(r'Standard Deviation ($\sigma_t$)')
ax.set_title('Gaussian Model Collapse: Simulation vs. Theory')
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('normal_collapse_comparison_plot.png')
plt.show()