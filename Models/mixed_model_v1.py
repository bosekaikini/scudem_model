import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
NUM_GENERATIONS = 2500  # Number of successive AI models to train
SAMPLE_SIZE = 1000    # The fixed size of the mixed training data set for each generation (M)
TRUE_MU = 0.0         # True mean of the original human-generated distribution
TRUE_SIGMA = 1.0      # True standard deviation of the original human-generated distribution
X_SYNTHETIC = 1.0     # Proportion (0.0 to 1.0) of data coming from the previous model.
                      # 1.0 = 100% Synthetic (max collapse)
                      # 0.0 = 0% Synthetic (no collapse)

def simulate_model_collapse(num_generations, sample_size, true_mu, true_sigma, x_synthetic):
    """
    Simulates model collapse using a single-dimensional Gaussian distribution with data mixing.

    The process models the paper's core mechanism:
    1. A model (Generation n) generates a synthetic sample set.
    2. A "real" (true distribution) sample set is generated.
    3. The next model (Generation n+1) is fitted on a mixture of these two sets.
    3. Error compounds over generations due to statistical approximation error.
    """
    
    # Calculate counts for mixing
    synthetic_count = int(sample_size * x_synthetic)
    real_count = sample_size - synthetic_count

    # Store results for plotting
    generation_history = []

    # Initialize Generation 0: The "True" Human Data Distribution
    current_mu = true_mu
    current_sigma = true_sigma
    
    generation_history.append({'mu': current_mu, 'sigma': current_sigma})

    print(f"--- Starting Simulation ({sample_size} samples/gen, {x_synthetic*100:.0f}% Synthetic) ---")
    print(f"Gen 0: Mu={current_mu:.4f}, Sigma={current_sigma:.4f}")

    # --- Recursive Training Loop (Generations 1 to N) ---
    for gen in range(1, num_generations + 1):
        
        # 1. GENERATE SYNTHETIC DATA (From the current model, Gen n)
        # This is where statistical approximation error is introduced due to finite sampling.
        synthetic_data = np.random.normal(loc=current_mu, scale=current_sigma, size=synthetic_count)
        
        # 2. GENERATE REAL DATA (From the original true distribution)
        # This acts as the stabilizing "human-generated" content.
        real_data = np.random.normal(loc=true_mu, scale=true_sigma, size=real_count)
        
        # 3. COMBINE DATASETS
        mixed_data = np.concatenate([synthetic_data, real_data])
        
        # 4. TRAIN NEXT MODEL (Fitting on the mixed data)
        # The next generation model is trained (fitted) on the mixed data.
        next_mu = np.mean(mixed_data)
        next_sigma = np.std(mixed_data)
        
        # Check for catastrophic collapse (variance goes to zero)
        if next_sigma < 1e-6:
             print(f"\n--- LATE COLLAPSE DETECTED at Generation {gen} ---")
             print("Variance has effectively collapsed to zero. Simulation halted.")
             current_mu = next_mu
             current_sigma = 0.0
             generation_history.append({'mu': current_mu, 'sigma': current_sigma})
             break

        # Update for the next iteration
        current_mu = next_mu
        current_sigma = next_sigma
        
        generation_history.append({'mu': current_mu, 'sigma': current_sigma})
        
        if gen % 5 == 0 or gen == num_generations:
            print(f"Gen {gen}: Mu={current_mu:.4f}, Sigma={current_sigma:.4f}")

    # --- Visualization ---
    
    generations = np.arange(len(generation_history))
    mus = np.array([g['mu'] for g in generation_history])
    sigmas = np.array([g['sigma'] for g in generation_history])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'Model Collapse Simulation ({x_synthetic*100:.0f}% Synthetic Data)', fontsize=16)

    # Plot 1: Standard Deviation (The primary indicator of collapse)
    ax1.plot(generations, sigmas, marker='o', linestyle='-', color='#1e90ff', label='Model $\sigma$')
    ax1.axhline(TRUE_SIGMA, color='red', linestyle='--', label=f'True $\sigma$ ({TRUE_SIGMA})')
    ax1.set_ylabel('Standard Deviation ($\sigma$)', fontsize=12)
    ax1.set_title('Loss of Diversity/Information (Variance Decay)', fontsize=14)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # Plot 2: Mean (Distribution Drift)
    ax2.plot(generations, mus, marker='s', linestyle='-', color='#3cb371', label='Model $\mu$')
    ax2.axhline(TRUE_MU, color='red', linestyle='--', label=f'True $\mu$ ({TRUE_MU})')
    ax2.set_xlabel('Model Generation (n)', fontsize=12)
    ax2.set_ylabel('Mean ($\mu$)', fontsize=12)
    ax2.set_title('Distribution Drift', fontsize=14)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Run the simulation
if __name__ == "__main__":
    # The X_SYNTHETIC variable in the configuration controls the proportion.
    simulate_model_collapse(NUM_GENERATIONS, SAMPLE_SIZE, TRUE_MU, TRUE_SIGMA, X_SYNTHETIC)

    # Try changing X_SYNTHETIC to see the effect:
    # X_SYNTHETIC = 1.0 (100% synthetic) -> Should collapse very quickly (like before)
    # X_SYNTHETIC = 0.5 (50% synthetic)  -> Should stabilize close to the true distribution
    # X_SYNTHETIC = 0.9 (90% synthetic)  -> Should collapse, but much slower