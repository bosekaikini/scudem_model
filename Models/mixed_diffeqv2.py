import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

NUM_GENERATIONS = 500
SAMPLE_SIZE = 50    
TRUE_SIGMA = 1.0   
X_SYNTHETIC = .95  

def variance_decay_solution(n, A, B, C):
    return A / np.sqrt(np.maximum(1.0, n + B)) + C

def simulate_model_collapse_and_fit_ode_solution(num_generations, sample_size, true_sigma, x_synthetic):
    
    synthetic_count = int(sample_size * x_synthetic)
    real_count = sample_size - synthetic_count

    sigma_history = []
    current_sigma = true_sigma
    sigma_history.append(current_sigma)

    print(f"--- Starting Simulation ({sample_size} samples/gen, {x_synthetic*100:.0f}% Synthetic) ---")

    for gen in range(1, num_generations + 1):
        
        synthetic_data = np.random.normal(loc=0.0, scale=current_sigma, size=synthetic_count)
        
        real_data = np.random.normal(loc=0.0, scale=true_sigma, size=real_count)
        
        mixed_data = np.concatenate([synthetic_data, real_data])
        next_sigma = np.std(mixed_data)
        
        if next_sigma < 1e-6:
             current_sigma = 0.0
             sigma_history.append(current_sigma)
             break

        current_sigma = next_sigma
        sigma_history.append(current_sigma)

    generations = np.arange(len(sigma_history))
    sigmas = np.array(sigma_history)
    
    fit_success = False
    popt = None
    try:
        popt, pcov = curve_fit(
            variance_decay_solution, 
            generations, 
            sigmas, 
            p0=[0.5, 1.0, 0.1], 
            maxfev=5000 
        )
        fitted_sigmas = variance_decay_solution(generations, *popt)
        fit_success = True
        
        A, B, C = popt
        print(f"\n--- Optimized Fit Parameters ---")
        print(f"Decay Amplitude (A): {A:.5f}")
        print(f"Initial Offset (B): {B:.3f}")
        print(f"Asymptotic Floor (C): {C:.5f}")
        print(f"Fitted Solution: sigma(n) = {A:.5f} / sqrt(n + {B:.3f}) + {C:.5f}")

    except RuntimeError:
        print("\nWarning: Curve fit failed. Plotting simulation only.")
    
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    title = 'Model Collapse: Standard Deviation vs. Improved Decay Model'
    fig.suptitle(title, fontsize=14)

    ax1.plot(generations, sigmas, marker='.', linestyle='', 
             color='#1e90ff', label='Simulation Trajectory', alpha=0.7)
    
    if fit_success:
        ode_label = r'Fit: $\sigma(n) = \frac{A}{\sqrt{n + B}} + C$'
        ax1.plot(generations, fitted_sigmas, color='red', linestyle='--', linewidth=2,
                 label=ode_label)
        ax1.set_title(f'Collapse Modelled with Floor $C={C:.5f}$', fontsize=12)

    ax1.axhline(TRUE_SIGMA, color='gray', linestyle=':', label=f'True $\sigma$ ({TRUE_SIGMA})')
    ax1.axhline(C, color='purple', linestyle=':', label=f'Fitted Floor (C={C:.5f})')
    ax1.set_xlabel('Model Generation (n)', fontsize=12)
    ax1.set_ylabel('Standard Deviation ($\sigma$)', fontsize=12)
    ax1.set_ylim(bottom=0) 
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    simulate_model_collapse_and_fit_ode_solution(NUM_GENERATIONS, SAMPLE_SIZE, TRUE_SIGMA, X_SYNTHETIC)
