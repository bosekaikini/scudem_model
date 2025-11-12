import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


np.random.seed(42)

num_generations = 100         
samples_per_gen = 200
eval_samples_per_gen = 2000  
eval_repeat = 20         
num_runs = 5        
mix_ratio = 0.0            

temperature = 0.75      
k_trunc = 2.0             
shrinkage = 0.995         
use_reinforce = True  
quantize_step = None         

def train_model(data):
    mean = np.mean(data)
    std = np.std(data, ddof=0) 
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
    var = model_std ** 2
    ll = -0.5 * np.log(2 * np.pi * var) - ((samples - model_mean) ** 2) / (2 * var)
    nll = -ll 
    return np.exp(nll)

true_mean = 0.0
true_std = 1.0
real_samples = np.random.normal(true_mean, true_std, size=2000)  
model0_mean, model0_std = train_model(real_samples)

print("Model0 (original) mean/std:", model0_mean, model0_std)

all_runs_hist_per_gen = []   
all_runs_mean_perp = []    
all_runs_std_perp = []     
all_runs_model_params = []   

for run in range(num_runs):
    np.random.seed(1000 + run) 
    current_data = np.random.choice(real_samples, size=samples_per_gen, replace=False)

    run_hist_per_gen = []
    run_mean_perp = []
    run_std_perp = []
    run_params = []

    for gen in range(num_generations + 1):
        m_mean, m_std = train_model(current_data)
        m_std = m_std * (shrinkage ** gen) 
        
        run_params.append((m_mean, m_std))

        per_sample_perps = []
        synth_for_eval = sample_from_model(m_mean, m_std, eval_samples_per_gen, temperature=1.0,
                                          k_trunc=None, quantize_step=None)
        perps = perplexity_of_samples_under_model(synth_for_eval, model0_mean, model0_std)
        run_hist_per_gen.append(perps) 
        run_mean_perp.append(np.mean(perps))
        run_std_perp.append(np.std(perps))

        if gen < num_generations:
            if use_reinforce:
                probs = gaussian_pdf(current_data, m_mean, m_std)
                if probs.sum() == 0:
                    probs = np.ones_like(probs) / len(probs)
                else:
                    probs = probs / probs.sum()
                next_train = np.random.choice(current_data, size=samples_per_gen, replace=True, p=probs)
                next_train = next_train + np.random.normal(0, m_std * temperature, size=samples_per_gen)
                low = m_mean - k_trunc * m_std
                high = m_mean + k_trunc * m_std
                next_train = np.clip(next_train, low, high)
            else:
                next_train = sample_from_model(m_mean, m_std, samples_per_gen,
                                               temperature=temperature, k_trunc=k_trunc,
                                               quantize_step=quantize_step)
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

all_runs_mean_perp = np.array(all_runs_mean_perp)  
all_runs_std_perp = np.array(all_runs_std_perp)
gens = np.arange(num_generations + 1)


selected_gens = [0, 1, 2, 5, num_generations]  
colors = plt.cm.viridis(np.linspace(0,1,len(selected_gens)))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
bins = np.logspace(np.log10(0.5), np.log10(np.max(all_runs_hist_per_gen[0][0]) + 1e-6 + 100), 80)
for i, g in enumerate(selected_gens):
    perps = all_runs_hist_per_gen[0][g]  
    plt.hist(perps, bins=bins, density=True, alpha=0.5, label=f"Gen {g}", color=colors[i])
plt.xscale('log')
plt.xlabel('Perplexity (evaluated under Model0)')
plt.ylabel('Probability density')
plt.title('Histogram of per-sample perplexities (run 0)')
plt.legend()

plt.subplot(1,2,2)
mean_of_runs = np.mean(all_runs_mean_perp, axis=0)
std_of_runs = np.std(all_runs_mean_perp, axis=0) 
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


plt.figure(figsize=(8,5))
for run in range(num_runs):
    y = np.array(all_runs_std_perp[run]) 
    positive_mask = y > 0
    if positive_mask.sum() < 3:
        continue
    x = gens[positive_mask]
    y_pos = y[positive_mask]
    logy = np.log(y_pos)
    coefs = np.polyfit(x, logy, 1)  
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

print("Per-run mean perplexities by generation (rows=runs, cols=gens):")
print(np.round(all_runs_mean_perp, 3))
print("Per-run std(perplex) by generation (rows=runs, cols=gens):")
print(np.round(all_runs_std_perp, 3))
