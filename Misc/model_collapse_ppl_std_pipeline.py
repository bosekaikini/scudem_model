# model_collapse_ppl_std_pipeline.py
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

# ----------------------------
# 0. User config (toy settings)
# ----------------------------
REF_CHECKPOINT = "distilgpt2"   # reference model (kept fixed for scoring)
BASE_GEN_CHECKPOINT = "distilgpt2"  # base generation model (we will fine-tune it iteratively)
tokenizer_name = "distilgpt2"
block_size = 64            # tokens per block (as in paper)
seqs_per_generation = 50  # M (keep small for toy)
num_generations = 20        # number of generations (toy)
num_runs = 2               # independent runs for std estimation (toy)
ft_epochs_per_generation = 1  # small for toy; paper uses more
learning_rate = 5e-5

# ----------------------------
# 1. Download & prepare WikiText-2
# ----------------------------
print("Loading WikiText-2...")
ds = load_dataset("wikitext", "wikitext-2-v1")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# ---------- Tokenizer + models safe setup (paste this) ----------
# from transformers import AutoTokenizer, AutoModelForCausalLM

TOKENIZER_NAME = "distilgpt2"   # change if needed
GEN_MODEL_NAME  = "distilgpt2"
REF_MODEL_NAME  = "distilgpt2"

# 1) create tokenizer and ensure pad token
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
# reuse eos as pad (no vocab change)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id
print("Using pad token:", tokenizer.pad_token, pad_token_id)

# load models AFTER tokenizer set
gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_NAME).to(device)
ref_model = AutoModelForCausalLM.from_pretrained(REF_MODEL_NAME).to(device)

# ensure model configs know pad id
gen_model.config.pad_token_id = pad_token_id
ref_model.config.pad_token_id = pad_token_id


def tokenize_lines(examples):
    return tokenizer(examples["text"], return_attention_mask=False)

# Build list of 64-token blocks from train split
print("Tokenizing and building blocks...")
train_texts = ds["train"]["text"]
all_ids = []
for txt in train_texts:
    if txt.strip() == "":
        continue
    ids = tokenizer(txt).input_ids
    all_ids.extend(ids)
# break into contiguous blocks of block_size
blocks = []
for i in range(0, len(all_ids) - block_size + 1, block_size):
    blocks.append(all_ids[i:i+block_size])
print("Built", len(blocks), f"{block_size}-token blocks.")

# ----------------------------
# 2. Utilities: compute perplexity for continuation only
# ----------------------------
@torch.no_grad()
def continuation_ppl(prompt_ids, generated_ids, ref_model):
    # prompt_ids, generated_ids: 1D lists/arrays of token ids
    full = torch.tensor(prompt_ids + generated_ids, dtype=torch.long, device=device).unsqueeze(0)
    input_ids = full[:, :-1]
    target_ids = full[:, 1:]
    outputs = ref_model(input_ids)
    logits = outputs.logits  # (1, L-1, V)
    log_probs = F.log_softmax(logits, dim=-1)  # natural log
    # only take target log-probs corresponding to generated tokens:
    P = len(prompt_ids)
    G = len(generated_ids)
    if G == 0:
        return float("nan")
    # slice positions P .. P+G-1 in target_ids
    gen_targets = target_ids[0, P:P+G]  # shape (G,)
    gen_log_probs = log_probs[0, P:P+G, :].gather(-1, gen_targets.unsqueeze(-1)).squeeze(-1)  # (G,)
    mean_log_prob = gen_log_probs.mean().item()
    nll = - mean_log_prob
    ppl = float(np.exp(nll))
    return ppl

# ----------------------------
# 3. Per-run / per-generation loop (toy fine-tune + generate + score)
# Note: heavy in real experiments. This is a toy minimal pipeline.
# ----------------------------
ref_model = AutoModelForCausalLM.from_pretrained(REF_CHECKPOINT).to(device)
ref_model.eval()


def fine_tune_model_on_dataset(base_checkpoint, train_blocks, output_dir, epochs=1, block_size=64):
    """
    Robust fine-tune helper.
    - Accepts train_blocks as either list[str] OR list[list[int]] (token ids).
    - Converts token-id lists to text via tokenizer.decode(...) before tokenizing in batch.
    """
    # 0. Ensure we have a fresh model instance
    model = AutoModelForCausalLM.from_pretrained(base_checkpoint).to(device)

    # 1. Normalize train_blocks to decoded strings
    normalized_texts = []
    for item in train_blocks:
        if isinstance(item, (list, tuple)):     # likely token id list
            normalized_texts.append(tokenizer.decode(item, clean_up_tokenization_spaces=True, skip_special_tokens=True))
        elif isinstance(item, str):
            normalized_texts.append(item)
        else:
            # fallback: cast to str
            normalized_texts.append(str(item))

    # 2. Build Dataset
    examples = {"text": normalized_texts}
    small_ds = Dataset.from_dict(examples)

    # 3. Tokenize in a safe way (handle different shapes)
    def tokenize_fn(examples):
        # examples["text"] is a list (batch)
        batch_texts = examples["text"]
        # sometimes HuggingFace returns nested lists if inputs were pre-tokenized; ensure strings
        if len(batch_texts) > 0 and isinstance(batch_texts[0], (list, tuple)):
            batch_texts = [tokenizer.decode(x, clean_up_tokenization_spaces=True, skip_special_tokens=True) for x in batch_texts]
        return tokenizer(batch_texts, truncation=True, max_length=block_size)

    tokenized = small_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    # sanity checks (run right after tokenized = small_ds.map(...))
    # compute tokenizer & model sizes and the max id present in the tokenized dataset
    vocab_len = len(tokenizer)  # current tokenizer length
    model_embed0 = model.get_input_embeddings().weight.size(0)
    # find max id in the dataset (may be slow but ok for debugging)
    max_id = -1
    for ex in tokenized["input_ids"]:
        if len(ex) > 0:
            mx = max(ex)
            if mx > max_id:
                max_id = mx

    print("DEBUG: tokenizer length =", vocab_len)
    print("DEBUG: model embedding rows =", model_embed0)
    print("DEBUG: max token id in tokenized dataset =", max_id)

    # 4. Data collator & training args
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,   # reduce if OOM
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


# storage: std_ppl_by_gen[run_idx][gen_idx] = std across sequences for that run? We'll follow the paper: std across *runs* at each generation.
# We'll store per-run per-gen mean & list of per-sequence ppls.
per_run_per_gen_ppls = [ [ [] for _ in range(num_generations) ] for _ in range(num_runs) ]

for run_idx in range(num_runs):
    print(f"\n=== RUN {run_idx+1}/{num_runs} ===")
    # start from base generator (fresh copy)
    gen_checkpoint = BASE_GEN_CHECKPOINT
    # Optionally, fine-tune generator once on real data to form "model0" similar to paper:
    # sample some real blocks for initial fine-tune (toy)
    real_init = random.sample(blocks, 500) if len(blocks) > 500 else blocks
    gen_model = fine_tune_model_on_dataset(gen_checkpoint, real_init, output_dir=f"./ft_run{run_idx}_gen0", epochs=1)
    gen_model.eval()

    for gen_idx in range(num_generations):
        print(f" generation {gen_idx} ...")
        # Generate sequences_per_generation continuations:
        seq_ppls = []
        # --- corrected generation + collection block (inside your per-generation loop) ---
        generated_training_texts = []  # will hold decoded strings (prompt+generated) for fine-tuning

        for s in range(seqs_per_generation):
            # pick a random block from real data
            prompt_block = random.choice(blocks)  # a list of token ids length block_size
            prompt_ids = prompt_block[: block_size // 2]  # list of ints

            # prepare input tensors (add batch dim); also build attention_mask = ones for the prompt
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids, device=device)

            # generate continuation (use model.generate with attention_mask)
            with torch.no_grad():
                out = gen_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=block_size // 2,
                    do_sample=True,
                    top_k=40,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id  # ensure pad token set
                )

            # out[0] is the full sequence (prompt + generated), as token ids
            full_ids = out[0].cpu().tolist()  # list of ints
            gen_ids = full_ids[len(prompt_ids):]  # generated token ids only

            # compute perplexity of the continuation under reference model
            ppl = continuation_ppl(prompt_ids, gen_ids, ref_model)
            seq_ppls.append(ppl)

            # Build the text to fine-tune on: decode prompt + generated into raw text (string)
            # Use tokenizer.decode on the concatenated id list; skip special tokens for cleanliness
            generated_text = tokenizer.decode(prompt_ids + gen_ids, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=True)
            generated_training_texts.append(generated_text)

        # After generating all sequences for this generation:
        per_run_per_gen_ppls[run_idx][gen_idx] = seq_ppls

        # Optionally fine-tune the generator on generated_training_texts
        if ft_epochs_per_generation > 0:
            # mix in a fraction gamma of original real blocks (if you want)
            gamma = 0.0  # set to 0.1 for 10% original data mixing as in paper
            n_orig = int(gamma * len(generated_training_texts))
            orig_texts_for_mix = []
            if n_orig > 0:
                sampled_blocks = random.sample(blocks, n_orig)
                orig_texts_for_mix = [tokenizer.decode(b, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                                      for b in sampled_blocks]

            # final fine-tune dataset (strings)
            train_texts_ft = generated_training_texts + orig_texts_for_mix
            # limit size for toy run to keep speed reasonable:
            train_texts_ft = train_texts_ft[:500]

            # call fine_tune_model_on_dataset(base_checkpoint, train_texts_ft, ...)
            gen_model = fine_tune_model_on_dataset(BASE_GEN_CHECKPOINT, train_texts_ft,
                                                   output_dir=f"./ft_run{run_idx}_gen{gen_idx + 1}",
                                                   epochs=ft_epochs_per_generation)
            gen_model.eval()

# ----------------------------
# 4. Aggregate across runs: compute mean & std of per-sequence perplexities at each generation
# ----------------------------
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

# ----------------------------
# 5. Fit algebraic model to std_ppl (or mean_ppl)
# ----------------------------
def fit_fn(t, A, B, C):
    return A * np.sqrt(1.0 / (1.0 + t / B)) + C

t = np.arange(num_gen)
y = std_ppl  # target metric
p0 = [float(y[0]) if not np.isnan(y[0]) else 1.0, seqs_per_generation, 0.0]
popt, pcov = curve_fit(fit_fn, t, y, p0=p0, maxfev=10000)
A_opt, B_opt, C_opt = popt
print("Fitted params (std):", A_opt, B_opt, C_opt)
k_est_from_fit = 1.0 / (2.0 * B_opt * A_opt * A_opt)
print("k from fitted params:", k_est_from_fit)

# ----------------------------
# 6. Validate ODE: finite difference derivative regression
# d/dt z = -k z^3  =>  estimate k by linear regression of (-dz/dt) vs z^3
# ----------------------------
z = y - C_opt
# central finite difference for dz/dt
dz_dt = np.zeros_like(z)
for i in range(len(z)):
    if i == 0:
        dz_dt[i] = (z[1] - z[0])
    elif i == len(z)-1:
        dz_dt[i] = (z[-1] - z[-2])
    else:
        dz_dt[i] = (z[i+1] - z[i-1]) / 2.0
# scale by 1 since dt=1 between generations; if dt !=1 use dt
lhs = -dz_dt  # should equal k * z^3
valid_mask = (~np.isnan(z)) & (~np.isnan(lhs))
# linear regression k_hat = sum(lhs*z^3)/sum((z^3)^2)
Z3 = (z**3)[valid_mask]
LHS = lhs[valid_mask]
if len(Z3) > 0 and np.sum(Z3**2) > 0:
    k_hat = np.sum(LHS * Z3) / np.sum(Z3**2)
else:
    k_hat = float('nan')
print("k hat from finite-diff regression:", k_hat)

# ----------------------------
# 7. Plot results
# ----------------------------
plt.figure(figsize=(8,5))
plt.plot(t, y, label='std_ppl (data)', linewidth=2)
plt.plot(t, fit_fn(t, *popt), '--', label='algebraic fit', color='red')
plt.fill_between(t, y - 0.0, y + 0.0, alpha=0.08)
plt.xlabel("Generation")
plt.ylabel("Std of Perplexity")
plt.title("Std(Perplexity) vs Generation (+ algebraic fit)")
plt.legend()
plt.show()

# Also plot -dz/dt vs z^3 scatter and regression line
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


# ----------------------------
# Save STD information
# ----------------------------

np.save("std_ppl.npy", np.asarray(std_ppl))          # binary NumPy file
# also save CSV for easy viewing
np.savetxt("std_ppl.csv", np.asarray(std_ppl), delimiter=",", header="std_ppl", comments='')
print("Wrote std_ppl.npy and std_ppl.csv")

generations = np.arange(num_generations)
std_ppl = np.array(std_ppl)  # or whatever name holds your STD values


# ----------------------------
# 8. See if square root/exponential fit is better
# ----------------------------

# --- assume std_ppl is an np.array or list available here ---


# pick the existing variable (try several common names)
sigma = np.asarray(std_ppl)

t = np.arange(len(sigma))

# algebraic fit function
def alg(t, A, B, C):
    return A * np.sqrt(1.0 / (1.0 + t / B)) + C

# exponential alternative
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

# Plot
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

# ---- defensive comparison block ----
print("Debug ranges before plotting:")
print("data min/max:", np.nanmin(sigma), np.nanmax(sigma))

# re-fit exponential with bounds to avoid explosion
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
# -------------------------------------



