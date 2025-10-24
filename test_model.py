import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  

mu_0 = 0
sigma_0 = 1

def generate_normal_samples(mu, sigma, n):
    '''
    mu: mean
    sigma: standard deviation
    n: number of samples
    '''
    return np.random.normal(mu, sigma, n)

def ml_estimate(samples):
    '''
    samples: array of samples
    '''
    # Note: ddof=1 is for unbiased variance, which is generally used for true MLE in this context
    # However, the paper's theory focuses on the MLE of sigma^2, which uses ddof=0. 
    # Since the simulation matches the paper, we'll keep the original function but 
    # recognize that np.std(..., ddof=0) is typically used for the maximum likelihood estimate
    return np.mean(samples), np.std(samples, ddof=1) 

n = 100  # Number of samples per generation
num_trajectories = 25   # Number of different trajectories to generate
num_generations = 2500  # Number of generations to simulate

means, stds = [], []
pbar = tqdm(total=num_trajectories * num_generations)

# Generate each trajectory
for _ in range(num_trajectories):
    mu, sigma = mu_0, sigma_0
    mean_arr, std_arr = [], [] 

    for _ in range(num_generations):
       # Generate samples and estimate the new probability
        samples = generate_normal_samples(mu, sigma, n)
        mu, sigma = ml_estimate(samples)
        mean_arr.append(mu)
        std_arr.append(sigma)
        pbar.update(1)

    means.append(mean_arr)
    stds.append(std_arr)

pbar.close()

# --- NEW CODE ADDED TO SAVE DATA ---

# Convert the list of lists (stds) into a NumPy array for easy saving
# Shape will be (num_trajectories, num_generations)
std_data_array = np.array(stds)

# Define the output file name
output_filename = 'gaussian_collapse_stds_data.csv'

# Save the array to a CSV file. Each row is one simulation trajectory.
np.savetxt(
    output_filename, 
    std_data_array, 
    delimiter=',', 
    header='Standard Deviation (sigma) for 25 trajectories over 2500 generations. Each row is a single trajectory.',
    comments='# '
)

print(f"\nSuccessfully saved standard deviation data to: {output_filename}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for mean_arr in means:
    ax[0].plot(mean_arr)
for std_arr in stds:
    ax[1].plot(std_arr)

ax[0].set_xlabel('Generation')
ax[1].set_xlabel('Generation')
ax[0].set_ylabel(r'Estimated Mean ($\mu$)')
ax[1].set_ylabel(r'Estimated Standard Deviation ($\sigma$)')
ax[0].set_title('Estimated Means')
ax[1].set_title('Estimated Standard Deviations')

plt.tight_layout()
plt.show()
