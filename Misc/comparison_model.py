import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



def theoretical_collapse_rate(t, n, sigma_0):
    return sigma_0 * np.sqrt(1 / (1 + t / n)) - 0.00008*t




mu_0 = 0
sigma_0 = 1 

def generate_normal_samples(mu, sigma, n):
    
    rng = np.random.default_rng() 
    return rng.normal(mu, sigma, n)

def ml_estimate(samples):
    
    return np.mean(samples), np.std(samples, ddof=1)

n = 100  
num_trajectories = 25   
num_generations = 2500 

means, stds = [], []
pbar = tqdm(total=num_trajectories * num_generations)


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


generations = np.arange(num_generations)

theoretical_stds = theoretical_collapse_rate(generations, n, sigma_0)



fig, ax = plt.subplots(1, 1, figsize=(10, 6))


ax.plot(
    generations,
    ensemble_average_stds, 
    color='blue', 
    label=f'Simulation Average ({num_trajectories} Trajectories)',
    linewidth=2
)


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
