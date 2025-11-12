import numpy as np
import matplotlib.pyplot as plt

NUM_GENERATIONS = 100 
SAMPLE_SIZE = 1000  
TRUE_MU = 0.0      
TRUE_SIGMA = 1.0    

def simulate_model_collapse(num_generations, sample_size, true_mu, true_sigma):
    generation_history = []

    current_mu = true_mu
    current_sigma = true_sigma
    
    generation_history.append({'mu': current_mu, 'sigma': current_sigma})

    print(f"--- Starting Simulation ({sample_size} samples per generation) ---")
    print(f"Gen 0: Mu={current_mu:.4f}, Sigma={current_sigma:.4f}")

    for gen in range(1, num_generations + 1):
        
        synthetic_data = np.random.normal(loc=current_mu, scale=current_sigma, size=sample_size)
        
        next_mu = np.mean(synthetic_data)
        next_sigma = np.std(synthetic_data)
        
        if next_sigma < 1e-6:
             print(f"\n--- LATE COLLAPSE DETECTED at Generation {gen} ---")
             print("Variance has effectively collapsed to zero. Simulation halted.")
             current_mu = next_mu
             current_sigma = 0.0
             generation_history.append({'mu': current_mu, 'sigma': current_sigma})
             break

        current_mu = next_mu
        current_sigma = next_sigma
        
        generation_history.append({'mu': current_mu, 'sigma': current_sigma})
        
        if gen % 5 == 0 or gen == num_generations:
            print(f"Gen {gen}: Mu={current_mu:.4f}, Sigma={current_sigma:.4f}")
    
    generations = np.arange(len(generation_history))
    mus = np.array([g['mu'] for g in generation_history])
    sigmas = np.array([g['sigma'] for g in generation_history])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'Model Collapse Simulation (N={sample_size} Samples/Gen)', fontsize=16)

    ax1.plot(generations, sigmas, marker='o', linestyle='-', color='#1e90ff', label='Model $\sigma$')
    ax1.axhline(TRUE_SIGMA, color='red', linestyle='--', label=f'True $\sigma$ ({TRUE_SIGMA})')
    ax1.set_ylabel('Standard Deviation ($\sigma$)', fontsize=12)
    ax1.set_title('Loss of Diversity/Information (Variance Decay)', fontsize=14)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    ax2.plot(generations, mus, marker='s', linestyle='-', color='#3cb371', label='Model $\mu$')
    ax2.axhline(TRUE_MU, color='red', linestyle='--', label=f'Initial $\mu$ ({TRUE_MU})')
    ax2.set_xlabel('Model Generation (n)', fontsize=12)
    ax2.set_ylabel('Mean ($\mu$)', fontsize=12)
    ax2.set_title('Distribution Drift', fontsize=14)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    simulate_model_collapse(NUM_GENERATIONS, SAMPLE_SIZE, TRUE_MU, TRUE_SIGMA)
