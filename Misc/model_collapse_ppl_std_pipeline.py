
import os, math, random, time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.metrics import r2_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


REF_CHECKPOINT = "distilgpt2"   
BASE_GEN_CHECKPOINT = "distilgpt2"  
tokenizer_name = "distilgpt2"
block_size = 64            
seqs_per_generation = 50  
num_generations = 20        
num_runs = 2               
ft_epochs_per_generation = 1 
learning_rate = 5e-5


print("Loading WikiText-2...")
ds = load_dataset("wikitext", "wikitext-2-v1")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)



TOKENIZER_NAME = "distilgpt2"  
GEN_MODEL_NAME  = "distilgpt2"
REF_MODEL_NAME  = "distilgpt2"


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id
print("Using pad token:", tokenizer.pad_token, pad_token_id)


gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_NAME).to(device)
ref_model = AutoModelForCausalLM.from_pretrained(REF_MODEL_NAME).to(device)


gen_model.config.pad_token_id = pad_token_id
ref_model.config.pad_token_id = pad_token_id


def tokenize_lines(examples):
    return tokenizer(examples["text"], return_attention_mask=False)


print("Tokenizing and building blocks...")
train_texts = ds["train"]["text"]
all_ids = []
for txt in train_texts:
    if txt.strip() == "":
        continue
    ids = tokenizer(txt).input_ids
    all_ids.extend(ids)

blocks = []
for i in range(0, len(all_ids) - block_size + 1, block_size):
    blocks.append(all_ids[i:i+block_size])
print("Built", len(blocks), f"{block_size}-token blocks.")


@torch.no_grad()
def continuation_ppl(prompt_ids, generated_ids, ref_model):
    
    full = torch.tensor(prompt_ids + generated_ids, dtype=torch.long, device=device).unsqueeze(0)
    input_ids = full[:, :-1]
    target_ids = full[:, 1:]
    outputs = ref_model(input_ids)
    logits = outputs.logits  
    log_probs = F.log_softmax(logits, dim=-1)  
    
    P = len(prompt_ids)
    G = len(generated_ids)
    if G == 0:
        return float("nan")
    
    gen_targets = target_ids[0, P:P+G]  
    gen_log_probs = log_probs[0, P:P+G, :].gather(-1, gen_targets.unsqueeze(-1)).squeeze(-1)  # (G,)
    mean_log_prob = gen_log_probs.mean().item()
    nll = - mean_log_prob
    ppl = float(np.exp(nll))
    return ppl


ref_model = AutoModelForCausalLM.from_pretrained(REF_CHECKPOINT).to(device)
ref_model.eval()


def fine_tune_model_on_dataset(base_checkpoint, train_blocks, output_dir, epochs=1, block_size=64):
    """
    Robust fine-tune helper.
    - Accepts train_blocks as either list[str] OR list[list[int]] (token ids).
    - Converts token-id lists to text via tokenizer.decode(...) before tokenizing in batch.
    """
    
    model = AutoModelForCausalLM.from_pretrained(base_checkpoint).to(device)

    
    normalized_texts = []
    for item in train_blocks:
        if isinstance(item, (list, tuple)):    
            normalized_texts.append(tokenizer.decode(item, clean_up_tokenization_spaces=True, skip_special_tokens=True))
        elif isinstance(item, str):
            normalized_texts.append(item)
        else:
            
            normalized_texts.append(str(item))

    
    examples = {"text": normalized_texts}
    small_ds = Dataset.from_dict(examples)

    
    def tokenize_fn(examples):
        
        batch_texts = examples["text"]
       
        if len(batch_texts) > 0 and isinstance(batch_texts[0], (list, tuple)):
            batch_texts = [tokenizer.decode(x, clean_up_tokenization_spaces=True, skip_special_tokens=True) for x in batch_texts]
        return tokenizer(batch_texts, truncation=True, max_length=block_size)

    tokenized = small_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    vocab_len = len(tokenizer)
    model_embed0 = model.get_input_embeddings().weight.size(0)
    
    max_id = -1
    for ex in tokenized["input_ids"]:
        if len(ex) > 0:
            mx = max(ex)
            if mx > max_id:
                max_id = mx

    print("DEBUG: tokenizer length =", vocab_len)
    print("DEBUG: model embedding rows =", model_embed0)
    print("DEBUG: max token id in tokenized dataset =", max_id)

    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=epochs,
        learning_rate=5e-5,
        logging_steps=50,
        save_strategy="no",
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=True,
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=data_collator)
    trainer.train()
    return model



per_run_per_gen_ppls = [ [ [] for _ in range(num_generations) ] for _ in range(num_runs) ]

for run_idx in range(num_runs):
    print(f"\n=== RUN {run_idx+1}/{num_runs} ===")
    
    gen_checkpoint = BASE_GEN_CHECKPOINT
   
    real_init = random.sample(blocks, 500) if len(blocks) > 500 else blocks
    gen_model = fine_tune_model_on_dataset(gen_checkpoint, real_init, output_dir=f"./ft_run{run_idx}_gen0", epochs=1)
    gen_model.eval()

    for gen_idx in range(num_generations):
        print(f" generation {gen_idx} ...")
        
        seq_ppls = []
        
        generated_training_texts = []

        for s in range(seqs_per_generation):
            
            prompt_block = random.choice(blocks)
            prompt_ids = prompt_block[: block_size // 2]

            
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids, device=device)

           
            with torch.no_grad():
                out = gen_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=block_size // 2,
                    do_sample=True,
                    top_k=40,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id 
                )

            
            full_ids = out[0].cpu().tolist()
            gen_ids = full_ids[len(prompt_ids):]

            
            ppl = continuation_ppl(prompt_ids, gen_ids, ref_model)
            seq_ppls.append(ppl)

            
            generated_text = tokenizer.decode(prompt_ids + gen_ids, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=True)
            generated_training_texts.append(generated_text)

        
        per_run_per_gen_ppls[run_idx][gen_idx] = seq_ppls

        
        if ft_epochs_per_generation > 0:
           
            gamma = 0.0 
            n_orig = int(gamma * len(generated_training_texts))
            orig_texts_for_mix = []
            if n_orig > 0:
                sampled_blocks = random.sample(blocks, n_orig)
                orig_texts_for_mix = [tokenizer.decode(b, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                                      for b in sampled_blocks]

            
            train_texts_ft = generated_training_texts + orig_texts_for_mix
            
            train_texts_ft = train_texts_ft[:500]

            
            gen_model = fine_tune_model_on_dataset(BASE_GEN_CHECKPOINT, train_texts_ft,
                                                   output_dir=f"./ft_run{run_idx}_gen{gen_idx + 1}",
                                                   epochs=ft_epochs_per_generation)
            gen_model.eval()


num_gen = num_generations
mean_ppl = np.zeros(num_gen)
std_ppl  = np.zeros(num_gen)

for g in range(num_gen):
    all_ppls = []
    for r in range(num_runs):
        all_ppls.extend([p for p in per_run_per_gen_ppls[r][g] if (p is not None and not np.isnan(p))])
    arr = np.array(all_ppls)
    mean_ppl[g] = np.nanmean(arr)
    std_ppl[g]  = np.nanstd(arr, ddof=1)


def fit_fn(t, A, B, C):
    return A * np.sqrt(1.0 / (1.0 + t / B)) + C

t = np.arange(num_gen)
y = std_ppl
p0 = [float(y[0]) if not np.isnan(y[0]) else 1.0, seqs_per_generation, 0.0]
popt, pcov = curve_fit(fit_fn, t, y, p0=p0, maxfev=10000)
A_opt, B_opt, C_opt = popt
print("Fitted params (std):", A_opt, B_opt, C_opt)
k_est_from_fit = 1.0 / (2.0 * B_opt * A_opt * A_opt)
print("k from fitted params:", k_est_from_fit)


z = y - C_opt

dz_dt = np.zeros_like(z)
for i in range(len(z)):
    if i == 0:
        dz_dt[i] = (z[1] - z[0])
    elif i == len(z)-1:
        dz_dt[i] = (z[-1] - z[-2])
    else:
        dz_dt[i] = (z[i+1] - z[i-1]) / 2.0

lhs = -dz_dt
valid_mask = (~np.isnan(z)) & (~np.isnan(lhs))

Z3 = (z**3)[valid_mask]
LHS = lhs[valid_mask]
if len(Z3) > 0 and np.sum(Z3**2) > 0:
    k_hat = np.sum(LHS * Z3) / np.sum(Z3**2)
else:
    k_hat = float('nan')
print("k hat from finite-diff regression:", k_hat)


plt.figure(figsize=(8,5))
plt.plot(t, y, label='std_ppl (data)', linewidth=2)
plt.plot(t, fit_fn(t, *popt), '--', label='algebraic fit', color='red')
plt.fill_between(t, y - 0.0, y + 0.0, alpha=0.08)
plt.xlabel("Generation")
plt.ylabel("Std of Perplexity")
plt.title("Std(Perplexity) vs Generation (+ algebraic fit)")
plt.legend()
plt.show()


plt.figure(figsize=(6,5))
plt.scatter(Z3, LHS[valid_mask], label='data')
if not math.isnan(k_hat):
    xs = np.linspace(min(Z3), max(Z3), 50)
    plt.plot(xs, k_hat*xs, color='red', label=f"best-fit k={k_hat:.4e}")
plt.xlabel("z^3 (where z = std - C)")
plt.ylabel("-dz/dt")
plt.legend()
plt.title("ODE validation: -dz/dt ≈ k * z^3")
plt.show()
print(f"σ(t) = {A_opt:.6f} * sqrt(1 / (1 + t/{B_opt:.6f})) + {C_opt:.6f}")
k = 1.0 / (2.0 * B_opt * A_opt**2)
print("Differential model: dσ/dt = -", k, " * (σ -", C_opt, ")**3")




np.save("std_ppl.npy", np.asarray(std_ppl))       

np.savetxt("std_ppl.csv", np.asarray(std_ppl), delimiter=",", header="std_ppl", comments='')
print("Wrote std_ppl.npy and std_ppl.csv")

generations = np.arange(num_generations)
std_ppl = np.array(std_ppl)








sigma = np.asarray(std_ppl)

t = np.arange(len(sigma))


def alg(t, A, B, C):
    return A * np.sqrt(1.0 / (1.0 + t / B)) + C


def expf(t, a, b, c):
    return a * np.exp(b * t) + c

p_alg, _ = curve_fit(alg, t, sigma, p0=[sigma[0], 50.0, sigma[-1]], maxfev=20000)
p_exp, _ = curve_fit(expf, t, sigma, p0=[sigma[0], -0.01, sigma[-1]], maxfev=20000)

y_alg = alg(t, *p_alg)
y_exp = expf(t, *p_exp)

r2_alg = r2_score(sigma, y_alg)
r2_exp = r2_score(sigma, y_exp)

print("ALGB FIT (A,B,C):", np.round(p_alg, 6), " R2:", r2_alg)
print("EXP  FIT (a,b,c):", np.round(p_exp, 6), " R2:", r2_exp)


plt.figure(figsize=(7,4))
plt.plot(t, sigma, 'o-', label='observed std_ppl')
plt.plot(t, y_alg, '--', label=f'algebraic fit (R2={r2_alg:.3f})')
plt.plot(t, y_exp, ':', label=f'exponential fit (R2={r2_exp:.3f})')
plt.xlabel('generation (t)')
plt.ylabel('std(perplexity)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print("Debug ranges before plotting:")
print("data min/max:", np.nanmin(sigma), np.nanmax(sigma))


low = [0.0, -1.0, -np.inf]
high = [np.inf, 0.0, np.inf]
try:
    p_exp, _ = curve_fit(expf, t, sigma, p0=[sigma[0], -0.01, sigma[-1]], bounds=(low, high), maxfev=20000)
except Exception as e:
    print("bounded exp fit failed:", e)
    p_exp, _ = curve_fit(expf, t, sigma, p0=[sigma[0], -0.01, sigma[-1]], maxfev=20000)
y_exp = expf(t, *p_exp)
r2_exp = r2_score(sigma, y_exp)

print("ALG params / R2:", p_alg, r2_alg)
print("EXP params / R2:", p_exp, r2_exp)
print("y_alg min/max:", np.nanmin(y_alg), np.nanmax(y_alg))
print("y_exp min/max:", np.nanmin(y_exp), np.nanmax(y_exp))

plt.figure(figsize=(8,5))
plt.plot(t, sigma, 'o-', label='observed std_ppl', linewidth=2, color='C0')
plt.plot(t, y_alg, '--', label=f'algebraic fit (R2={r2_alg:.3f})', color='red', linewidth=2)
plt.plot(t, y_exp, ':', label=f'exponential fit (R2={r2_exp:.3f})', color='black', linewidth=2)
all_vals = np.concatenate([sigma, y_alg, y_exp])
plt.ylim(np.nanmin(all_vals) - 0.05*(np.nanmax(all_vals)-np.nanmin(all_vals)),
         np.nanmax(all_vals) + 0.05*(np.nanmax(all_vals)-np.nanmin(all_vals)))
plt.xlabel('generation (t)')
plt.ylabel('std(perplexity)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




