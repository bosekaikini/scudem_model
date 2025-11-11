"""
Recursive-generation numeric simulation (Gaussian analogue of "model collapse")
- Produces histograms of per-sample perplexities evaluated by Model0 (original).
- Tracks mean & std of perplexity per generation across runs.
- Fits an exponential (differential) model to std(perplexity) vs generation.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# ------------------------
# CONFIG
# ------------------------
np.random.seed(42)

num_generations = 100          # number of recursive generations to simulate (0..num_generations)
samples_per_gen = 200         # training sample size used to fit each generation-model
eval_samples_per_gen = 2000   # how many synthetic points to compute per-sample perplexities for histograms
eval_repeat = 20              # how many sub-samples to compute sample-wise perplexity statistics
num_runs = 5                  # independent repeats (like paper's runs)
mix_ratio = 0.0               # fraction of original data mixed into each generation (0.0 = none)
# collapse-inducing knobs (combine to cause collapse)
temperature = 0.75            # <1 concentrates sampling around mean (like low-temp)
k_trunc = 2.0                 # truncation threshold in std units (clip tails at mean ± k_trunc*std)
shrinkage = 0.995             # multiplicative shrinkage applied to learned std each generation
use_reinforce = True          # resample training data with probabilities proportional to current density
quantize_step = None          # None or step size (e.g., 0.05) to quantize outputs (optional)

# ------------------------
# HELPERS
# ------------------------
def train_model(data):
    mean = np.mean(data)
    std = np.std(data, ddof=0)  # MLE-style
    return mean, std

def gaussian_pdf(x, mean, std):
    var = std * std
    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * (x - mean) ** 2 / var)

def sample_from_model(mean, std, n, temperature=1.0, k_trunc=None, quantize_step=None):
    eff_std = std * temperature
    out = np.random.normal(mean, eff_std, size=n)
    if k_trunc is not None:
        low = mean - k_trunc * std
        high = mean + k_trunc * std
        out = np.clip(out, low, high)
    if quantize_step is not None:
        out = np.round(out / quantize_step) * quantize_step
    return out

def perplexity_of_samples_under_model(samples, model_mean, model_std):
    # negative log-likelihood per datum
    var = model_std ** 2
    ll = -0.5 * np.log(2 * np.pi * var) - ((samples - model_mean) ** 2) / (2 * var)
    nll = -ll  # per-sample negative log-likelihood
    # convert to perplexity-like metric per sample -> exp(nll)
    # NOTE: perplexity is often defined at sequence-level; this is sample-level analogue
    return np.exp(nll)

# ------------------------
# PREPARE "REAL" DATA and Model 0
# ------------------------
true_mean = 0.0
true_std = 1.0
real_samples = np.random.normal(true_mean, true_std, size=2000)  # the 'human' data
model0_mean, model0_std = train_model(real_samples)

print("Model0 (original) mean/std:", model0_mean, model0_std)

# ------------------------
# RUN MULTIPLE INDEPENDENT TRIALS
# ------------------------
all_runs_hist_per_gen = []   # store per-sample perplexities distributions evaluated under model0
all_runs_mean_perp = []      # shape (num_runs, num_generations+1)
all_runs_std_perp = []       # shape (num_runs, num_generations+1)
all_runs_model_params = []   # (means,stds) per gen per run

for run in range(num_runs):
    np.random.seed(1000 + run)  # reproducible per-run seeds
    # start training data = original data OR small sample drawn from original
    current_data = np.random.choice(real_samples, size=samples_per_gen, replace=False)

    run_hist_per_gen = []
    run_mean_perp = []
    run_std_perp = []
    run_params = []

    for gen in range(num_generations + 1):  # include generation 0
        # Train a model on current_data
        m_mean, m_std = train_model(current_data)
        # apply explicit shrinkage bias on estimated std (to simulate capacity/underestimation)
        m_std = m_std * (shrinkage ** gen)  # optionally stronger over time; here multiplicative by gen
        
        run_params.append((m_mean, m_std))

        # Evaluate: compute many samples from current_data and compute their perplexities under Model0
        # We'll take several repeated subsamples to compute stable mean/std of per-sample perplexity
        per_sample_perps = []
        # we choose synthetic evaluation data sampled from the *current generation's distribution*
        # (you could instead use a single large set from model sampling; here do multiple draws)
        synth_for_eval = sample_from_model(m_mean, m_std, eval_samples_per_gen, temperature=1.0,
                                          k_trunc=None, quantize_step=None)
        # compute per-sample perplexities under the ORIGINAL model (model0)
        perps = perplexity_of_samples_under_model(synth_for_eval, model0_mean, model0_std)
        run_hist_per_gen.append(perps)  # store distribution for left-panel histograms
        run_mean_perp.append(np.mean(perps))
        run_std_perp.append(np.std(perps))

        # Prepare training set for next generation (unless we're at final gen)
        if gen < num_generations:
            # Option A: self-reinforcement: resample current_data proportional to density under current model
            if use_reinforce:
                # density scores of points in current_data under current model (use gaussian_pdf)
                probs = gaussian_pdf(current_data, m_mean, m_std)
                if probs.sum() == 0:
                    probs = np.ones_like(probs) / len(probs)
                else:
                    probs = probs / probs.sum()
                next_train = np.random.choice(current_data, size=samples_per_gen, replace=True, p=probs)
                # Add temperature noise to simulate sampling from model (small noise)
                next_train = next_train + np.random.normal(0, m_std * temperature, size=samples_per_gen)
                # Truncate tails
                low = m_mean - k_trunc * m_std
                high = m_mean + k_trunc * m_std
                next_train = np.clip(next_train, low, high)
            else:
                # plain sample-from-model
                next_train = sample_from_model(m_mean, m_std, samples_per_gen,
                                               temperature=temperature, k_trunc=k_trunc,
                                               quantize_step=quantize_step)
            # optionally mix some original data to slow collapse
            if mix_ratio > 0.0:
                mix_n = int(mix_ratio * samples_per_gen)
                if mix_n > 0:
                    mix_pts = np.random.choice(real_samples, size=mix_n, replace=False)
                    next_train[:mix_n] = mix_pts
            current_data = next_train

    all_runs_hist_per_gen.append(run_hist_per_gen)
    all_runs_mean_perp.append(run_mean_perp)
    all_runs_std_perp.append(run_std_perp)
    all_runs_model_params.append(run_params)

# Convert to arrays for easier plotting
all_runs_mean_perp = np.array(all_runs_mean_perp)   # shape (runs, gens+1)
all_runs_std_perp = np.array(all_runs_std_perp)
gens = np.arange(num_generations + 1)

# ------------------------
# PLOT: HISTOGRAMS (left panels) - overlay several generation distributions evaluated under Model0
# ------------------------
selected_gens = [0, 1, 2, 5, num_generations]  # which generations to overlay (like paper)
colors = plt.cm.viridis(np.linspace(0,1,len(selected_gens)))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
# log-x bins (perplexity often over many orders)
bins = np.logspace(np.log10(0.5), np.log10(np.max(all_runs_hist_per_gen[0][0]) + 1e-6 + 100), 80)
for i, g in enumerate(selected_gens):
    # collect from run 0 for the histogram demonstration (paper shows many colors per generation)
    perps = all_runs_hist_per_gen[0][g]  # use run 0 to make left-panel histograms like paper
    plt.hist(perps, bins=bins, density=True, alpha=0.5, label=f"Gen {g}", color=colors[i])
plt.xscale('log')
plt.xlabel('Perplexity (evaluated under Model0)')
plt.ylabel('Probability density')
plt.title('Histogram of per-sample perplexities (run 0)')
plt.legend()

# ------------------------
# PLOT: Mean perplexity ± std across generations (right panel)
# ------------------------
plt.subplot(1,2,2)
mean_of_runs = np.mean(all_runs_mean_perp, axis=0)
std_of_runs = np.std(all_runs_mean_perp, axis=0)  # variation across runs of the mean-perplexity
# also compute average std-perplexity per generation across runs
avg_std_perp = np.mean(all_runs_std_perp, axis=0)
std_std_perp = np.std(all_runs_std_perp, axis=0)

plt.errorbar(gens, mean_of_runs, yerr=std_of_runs, marker='o', label='Mean Perplexity ± across runs')
plt.fill_between(gens, avg_std_perp - std_std_perp, avg_std_perp + std_std_perp,
                 color='orange', alpha=0.2, label='Std(per-sample-perplexity) mean ± sd')
plt.xlabel('Generation')
plt.ylabel('Perplexity')
plt.title('Mean perplexity (left) and std(per-sample-perplexity) (band)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ------------------------
# PLOT: Parameter drift (mean and std of generation models)
# ------------------------
plt.figure(figsize=(10,4))
for run in range(num_runs):
    params = np.array(all_runs_model_params[run])
    means_run = params[:,0]
    stds_run = params[:,1]
    plt.plot(gens, means_run, alpha=0.6, label=f'run{run} mean')
plt.hlines(model0_mean, 0, num_generations, colors='k', linestyles='--', label='Model0 mean')
plt.xlabel('Generation'); plt.ylabel('Learned mean'); plt.title('Learned means by generation'); plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,4))
for run in range(num_runs):
    params = np.array(all_runs_model_params[run])
    stds_run = params[:,1]
    plt.plot(gens, stds_run, alpha=0.6, label=f'run{run} std')
plt.hlines(model0_std, 0, num_generations, colors='k', linestyles='--', label='Model0 std')
plt.xlabel('Generation'); plt.ylabel('Learned std'); plt.title('Learned stds by generation'); plt.legend()
plt.grid(True)
plt.show()

# ------------------------
# DIFFERENTIAL (exponential) FIT TO std(perplexity) vs generation
# Fit log(std_perp) = log(sigma0) - k * t  => linear fit to get k
# ------------------------
plt.figure(figsize=(8,5))
for run in range(num_runs):
    y = np.array(all_runs_std_perp[run])  # std(per-sample-perplexity) for this run across generations
    # avoid zeros or negatives
    positive_mask = y > 0
    if positive_mask.sum() < 3:
        continue
    x = gens[positive_mask]
    y_pos = y[positive_mask]
    # linear fit on log scale
    logy = np.log(y_pos)
    coefs = np.polyfit(x, logy, 1)  # logy = a*x + b where a = -k
    a, b = coefs[0], coefs[1]
    k_est = -a
    yfit = np.exp(b + a * x)
    plt.plot(gens, y, 'o-', alpha=0.6, label=f'run{run} std_perp (k≈{k_est:.4f})')
    plt.plot(x, yfit, '--', color='C{}'.format(run), alpha=0.8)
plt.yscale('log')
plt.xlabel('Generation'); plt.ylabel('Std(perplexity) (log scale)')
plt.title('Exponential (ODE) fit to std(perplexity) vs generation')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.show()

# ------------------------
# ADDITIONAL DIAGNOSTICS: show per-run mean/std arrays
# ------------------------
print("Per-run mean perplexities by generation (rows=runs, cols=gens):")
print(np.round(all_runs_mean_perp, 3))
print("Per-run std(perplex) by generation (rows=runs, cols=gens):")
print(np.round(all_runs_std_perp, 3))

# =============================================================================
# Notes:
# - Tweak knobs (temperature, k_trunc, use_reinforce, shrinkage, samples_per_gen) to produce
#   stronger or weaker collapse. In particular, lowering temperature, stronger truncation (smaller k_trunc),
#   smaller samples_per_gen, and stronger shrinkage all accelerate shrinkage of variance.
# - The "perplexity" here is a sample-level analogue: perplexity = exp(-log p(x)) for single continuous datum.
# - The differential fit is a simple exponential fit (soln of dσ/dt = -k σ). You can alter to more
#   complex ODEs (logistic, multi-component) if the dynamics require it.
# =============================================================================
