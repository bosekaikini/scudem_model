import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os 

# --- New Function to Calculate and Save Ensemble Average ---

def plot_and_save_ensemble_average(stds_data_array, num_trajectories):
    """
    Calculates the ensemble average of standard deviations across all trajectories,
    plots the result, and saves the data as a single column CSV file.
    
    stds_data_array: 2D NumPy array of standard deviations (trajectories x generations)
    num_trajectories: The number of trajectories used for the average
    """
    
    # Calculate the Ensemble Average (Mean)
    # axis=0 averages down the columns, giving one average value per generation.
    ensemble_average_stds = np.mean(stds_data_array, axis=0)
    
    # --- Plotting the Ensemble Average ---
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(ensemble_average_stds, color='red', label=f'Ensemble Average ({num_trajectories} Trajectories)')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel(r'Ensemble Avg. Standard Deviation ($\bar{\sigma}$)')
    ax.set_title(f'Rate of Gaussian Model Collapse (Ensemble Average)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('normal_collapse_average_plot.png')
    plt.show()

    # --- Saving Averaged Data to CSV (One Column) ---
    
    avg_output_filename = 'gaussian_collapse_stds_AVERAGE_column.csv'
    
    # np.savetxt automatically saves 1D arrays as a single column.
    np.savetxt(
        avg_output_filename, 
        ensemble_average_stds, 
        delimiter='\n', # Use newline as delimiter for a single column output
        header=f'Ensemble Average Standard Deviation (sigma) across {num_trajectories} trajectories. (One value per row/generation)',
        comments='# '
    )
    print(f"\nSuccessfully saved ensemble average (single column) to: {avg_output_filename}")


# --- ORIGINAL SIMULATION CODE ---

# Initial parameters
mu_0 = 0
sigma_0 = 1

def generate_normal_samples(mu, sigma, n):
    '''
    mu: mean
    sigma: standard deviation
    n: number of samples
    '''
    # We must use np.random.default_rng() or similar to ensure it works outside a Colab cell
    rng = np.random.default_rng()
    return rng.normal(mu, sigma, n)

def ml_estimate(samples):
    '''
    samples: array of samples
    '''
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

# ----------------------------------------------------
# --- EXECUTION ---
# ----------------------------------------------------

# Convert the list of lists (stds) into a 2D NumPy array
std_data_array = np.array(stds)

# Call the new function to plot the average and save the single-column CSV
plot_and_save_ensemble_average(std_data_array, num_trajectories)