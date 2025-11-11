import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Configuration ---
NUM_GENERATIONS = 500 # Generations to show full collapse trend
SAMPLE_SIZE = 50      # Reduced sample size to accelerate collapse
TRUE_SIGMA = 1.0      # True standard deviation (sigma_0)
X_SYNTHETIC = 1.0     # 100% Synthetic data for fast collapse

# --- ODE Solution Function (What we are plotting and fitting) ---
def variance_decay_solution(n, A, B):
    """
    The solution function for the differential equation:
    d(sigma^2)/dn = -A * (sigma^2)^2 
    
    The solution is: sigma(n) = sqrt( 1 / (A*n + B) )
    We fit the simulation's standard deviation (sigma) to this function.
    """
    # Use np.maximum(1.0, n + B) to avoid division by zero or complex numbers
    # We must enforce non-negativity under the square root
    denominator = np.maximum(1e-9, A * n + B) 
    return np.sqrt(1 / denominator)

def simulate_model_collapse_and_fit_ode_solution(num_generations, sample_size, true_sigma, x_synthetic):
    """
    Simulates model collapse and fits the standard deviation to the ODE solution.
    """
    
    synthetic_count = int(sample_size * x_synthetic)
    real_count = sample_size - synthetic_count

    # Store standard deviation (sigma) history
    sigma_history = []
    current_sigma = true_sigma
    sigma_history.append(current_sigma)

    print(f"--- Starting Simulation ({sample_size} samples/gen, {x_synthetic*100:.0f}% Synthetic) ---")

    # --- Recursive Training Loop (Simulation) ---
    for gen in range(1, num_generations + 1):
        
        # 1. GENERATE SYNTHETIC DATA
        synthetic_data = np.random.normal(loc=0.0, scale=current_sigma, size=synthetic_count)
        
        # 2. GENERATE REAL DATA (Kept in loop for general case, but count is 0 if X_SYNTHETIC=1.0)
        real_data = np.random.normal(loc=0.0, scale=true_sigma, size=real_count)
        
        # 3. COMBINE & TRAIN
        mixed_data = np.concatenate([synthetic_data, real_data])
        next_sigma = np.std(mixed_data)
        
        if next_sigma < 1e-6:
             current_sigma = 0.0
             sigma_history.append(current_sigma)
             break

        current_sigma = next_sigma
        sigma_history.append(current_sigma)

    # --- Data Extraction and Curve Fitting (Modeling the Solution) ---
    generations = np.arange(len(sigma_history))
    sigmas = np.array(sigma_history)
    
    fit_success = False
    popt = None
    try:
        # Fit sigma to the solution function: sqrt( 1 / (A*n + B) )
        # Initial guess: B must be close to 1.0 (since sigma(0)=1.0, and sigma(0)=1/sqrt(B))
        popt, pcov = curve_fit(
            variance_decay_solution, 
            generations, 
            sigmas, 
            p0=[0.001, 1.0], # A (rate), B (initial condition)
            maxfev=5000 
        )
        fitted_sigmas = variance_decay_solution(generations, *popt)
        fit_success = True
        
        A, B = popt
        print(f"\n--- Optimized Fit Parameters ---")
        print(f"Differential Equation: d(sigma^2)/dn = -({A:.5f}) * (sigma^2)^2")
        print(f"Fitted Solution: sigma(n) = sqrt( 1 / ({A:.5f}*n + {B:.3f}) )")

    except RuntimeError:
        print("\nWarning: Curve fit failed. Plotting simulation only.")
    
    
    # --- Visualization (Plotting the Variable and the Solution) ---
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    title = 'Model Collapse: Standard Deviation vs. Differential Equation Solution'
    fig.suptitle(title, fontsize=14)

    # Plot 1: Simulation Data (The variable over time)
    ax1.plot(generations, sigmas, marker='.', linestyle='', 
             color='#1e90ff', label='Simulation Trajectory', alpha=0.7)
    
    if fit_success:
        # Plot 2: ODE Solution (The smooth modeling curve)
        ode_label = r'ODE Solution Fit: $\sigma(n) = \sqrt{1 / (An + B)}$'
        ax1.plot(generations, fitted_sigmas, color='red', linestyle='--', linewidth=2,
                 label=ode_label)
        ax1.set_title(f'Collapse Modelled by $d\sigma^2/dn = -A(\sigma^2)^2$ (A={A:.5f})', fontsize=12)

    ax1.axhline(TRUE_SIGMA, color='gray', linestyle=':', label=f'True $\sigma$ ({TRUE_SIGMA})')
    ax1.set_xlabel('Model Generation (n)', fontsize=12)
    ax1.set_ylabel('Standard Deviation ($\sigma$)', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Run the simulation
if __name__ == "__main__":
    simulate_model_collapse_and_fit_ode_solution(NUM_GENERATIONS, SAMPLE_SIZE, TRUE_SIGMA, X_SYNTHETIC)