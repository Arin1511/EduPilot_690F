#!/usr/bin/env python3
"""
distilgpt2_exposure_mia_zscore.py

Stronger exposure-style MIA using a decoder-only LM (DistilGPT2).

Key ideas:
- Decoder-only often memorizes tiny datasets better than encoder–decoder -> stronger exposure signal.
- Train on concatenated PROMPT + TARGET; loss only over TARGET tokens.
- Score with per-token AVG log-likelihood over TARGET.
- Use class-conditioned (job_role) negatives, K samples per example -> background mean+std.
- Final statistic: Z-SCORE (LiRA-style), typically sharper than a single-gap.

Outputs:
- exposure_results_gpt2.csv
- exposure_histograms_gpt2.png
- exposure_roc_curve_gpt2.png

Requires:
  torch, transformers (>=4.30 recommended), datasets, scikit-learn, pandas, matplotlib, numpy, accelerate (>=0.26)
"""

import os, sys, random
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
DATA_PATH = "EduPilot_dataset.csv"   # adjust path if needed
MODEL_NAME = "distilgpt2"
MODEL_DIR = "./distilgpt2_finetuned"

RANDOM_SEED = 42
NUM_EPOCHS = 40          # encourage mild overfit on tiny data
PER_DEVICE_BATCH = 8
LEARNING_RATE = 2e-4     # a touch higher for tiny data
WEIGHT_DECAY = 0.0
MAX_LEN = 256            # prompt + target packed
K_NEG = 8                # negatives per example for background stats

# -------------------------
# Dependency check
# -------------------------
missing = []
try: import torch
except Exception: missing.append("torch")
try: import transformers
except Exception: missing.append("transformers")
try: import datasets
except Exception: missing.append("datasets")
try: import sklearn
except Exception: missing.append("scikit-learn")
try: import matplotlib
except Exception: missing.append("matplotlib")
if missing:
    print("Missing:", ", ".join(missing))
    print("Install e.g.: pip install -U torch transformers datasets scikit-learn matplotlib 'accelerate>=0.26.0'")
    sys.exit(1)

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve, auc as calc_auc

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# -------------------------
# Helpers
# -------------------------
def make_prompt(row: dict) -> str:
    # Structured prompt reduces ambiguity; keep consistent format
    return (
        "task: generate interview mock question\n"
        f"query: {row.get('user_query','')}\n"
        f"role: {row.get('job_role','')}\n"
        f"company: {row.get('company','')}\n"
        f"location: {row.get('location','')}\n"
        f"round: {row.get('interview_round','')}\n"
        "target:"
    )

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    df["prompt"] = df.apply(make_prompt, axis=1)
    df["target"] = df["mock_question"].astype(str).str.strip()
    df = df[df["target"] != ""].reset_index(drop=True)
    return df

def stratified_split(df: pd.DataFrame, label_col="job_role", test_size=0.30, seed=RANDOM_SEED):
    labels = df[label_col].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(sss.split(df, labels))
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)

class PromptTargetCausalDataset(TorchDataset):
    """
    For each row, builds input_ids = [PROMPT + " " + TARGET + eos]
    Labels = input_ids with prompt tokens masked to -100 so loss applies only to TARGET.
    Also returns indices of target token positions for scoring.
    """
    def __init__(self, df: pd.DataFrame, tokenizer, max_len=MAX_LEN, add_eos=True):
        self.df = df
        self.tok = tokenizer
        self.max_len = max_len
        self.add_eos = add_eos
        self.eos = tokenizer.eos_token or ""
        # Pre-tokenize prompts to detect lengths quickly
        self.prompt_ids = []
        self.target_texts = []
        for _, r in df.iterrows():
            p = r["prompt"]
            t = r["target"]
            if add_eos and not t.endswith(self.eos):
                t = t + " " + self.eos
            self.target_texts.append(t)
            p_ids = self.tok(p, add_special_tokens=False)["input_ids"]
            self.prompt_ids.append(p_ids)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        p_ids = self.prompt_ids[idx]
        t_txt = self.target_texts[idx]
        t_ids = self.tok(t_txt, add_special_tokens=False)["input_ids"]

        # Build sequence: [prompt] + [space] + [target]
        space = self.tok(" ", add_special_tokens=False)["input_ids"]
        ids = p_ids + space + t_ids
        ids = ids[: self.max_len]
        attn = [1] * len(ids)

        # mask prompt tokens + space to -100
        prompt_len = min(len(p_ids) + len(space), len(ids))
        labels = [-100] * prompt_len + ids[prompt_len:]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            # for convenience during scoring:
            "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
        }

def compute_target_avg_logprob(
    model, tokenizer, prompts: List[str], targets: List[str], batch_size=8, max_len=MAX_LEN
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-token average log-prob over TARGET tokens only.
    Returns (sum_logp, tok_counts) to allow avg outside.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )
    model.to(device)
    model.eval()
    all_sum = []
    all_cnt = []
    ce = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]

            # Build batch the same way as training
            batch_ids = []
            batch_labels = []
            batch_attn = []
            for p, t in zip(batch_prompts, batch_targets):
                if tokenizer.eos_token and not t.endswith(tokenizer.eos_token):
                    t = t + " " + tokenizer.eos_token
                p_ids = tokenizer(p, add_special_tokens=False)["input_ids"]
                t_ids = tokenizer(t, add_special_tokens=False)["input_ids"]
                space = tokenizer(" ", add_special_tokens=False)["input_ids"]
                ids = (p_ids + space + t_ids)[:max_len]
                attn = [1]*len(ids)
                prompt_len = min(len(p_ids)+len(space), len(ids))
                labels = [-100]*prompt_len + ids[prompt_len:]
                batch_ids.append(ids); batch_attn.append(attn); batch_labels.append(labels)

            input_ids = torch.tensor([x + [tokenizer.pad_token_id]*(max_len-len(x)) for x in batch_ids], dtype=torch.long).to(device)
            attention_mask = torch.tensor([x + [0]*(max_len-len(x)) for x in batch_attn], dtype=torch.long).to(device)
            labels = torch.tensor([x + [-100]*(max_len-len(x)) for x in batch_labels], dtype=torch.long).to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[:, :-1, :]            # shift for teacher-forcing alignment
            shift_labels = labels[:, 1:]

            loss_tok = ce(logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1))
            loss_tok = loss_tok.view(shift_labels.size(0), shift_labels.size(1))

            # Mask only target tokens (where label != -100)
            mask = (shift_labels != -100).float()
            tok_counts = mask.sum(dim=1).cpu().numpy()
            sum_logp = -(loss_tok * mask).sum(dim=1).cpu().numpy()

            all_sum.extend(sum_logp.tolist())
            all_cnt.extend(tok_counts.tolist())

    return np.array(all_sum), np.array(all_cnt)

# -------------------------
# Load + split
# -------------------------
df = load_dataset(DATA_PATH)
print(f"Loaded dataset: {len(df)} rows")
train_df, test_df = stratified_split(df, label_col="job_role", test_size=0.30, seed=RANDOM_SEED)
print(f"Train: {len(train_df)}  Test: {len(test_df)}")

# Buckets by job_role for background negatives
bucket_key = "job_role"
buckets = {}
for _, r in df.iterrows():
    k = str(r.get(bucket_key, "UNK"))
    buckets.setdefault(k, []).append(str(r["target"]))

# -------------------------
# Tokenizer + model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# ensure EOS exists
if tokenizer.eos_token is None:
    tokenizer.eos_token = "</eos>"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
# reduce/drop dropout to help memorization on tiny data
try:
    if hasattr(model.config, "attn_pdrop"): model.config.attn_pdrop = 0.0
    if hasattr(model.config, "embd_pdrop"): model.config.embd_pdrop = 0.0
    if hasattr(model.config, "resid_pdrop"): model.config.resid_pdrop = 0.0
except Exception:
    pass

# -------------------------
# Torch dataset for training
# -------------------------
class TrainSet(PromptTargetCausalDataset): ...
train_ds = TrainSet(train_df, tokenizer, max_len=MAX_LEN, add_eos=True)

# Collator: we already supply labels, so just pad to max in-batch
def collate_fn(batch):
    max_len_b = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for x in batch:
        L = len(x["input_ids"])
        pad = max_len_b - L
        input_ids.append(x["input_ids"].tolist() + [tokenizer.pad_token_id]*pad)
        attention_mask.append(x["attention_mask"].tolist() + [0]*pad)
        labels.append(x["labels"].tolist() + [-100]*pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

# -------------------------
# Train
# -------------------------
args = TrainingArguments(
    output_dir=MODEL_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    logging_steps=10,
    seed=RANDOM_SEED
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=collate_fn,
    train_dataset=train_ds,
    tokenizer=tokenizer
)

print("Starting fine-tuning (DistilGPT2)...")
trainer.train()
trainer.save_model(MODEL_DIR)
print(f"Model saved to {MODEL_DIR}")

# -------------------------
# Compute true per-token avg log-likelihood (target only)
# -------------------------
train_prompts = train_df["prompt"].tolist()
train_targets = train_df["target"].tolist()
test_prompts  = test_df["prompt"].tolist()
test_targets  = test_df["target"].tolist()

print("Scoring true log-probs (train)...")
tr_sum, tr_cnt = compute_target_avg_logprob(model, tokenizer, train_prompts, train_targets, batch_size=4)
print("Scoring true log-probs (test)...")
te_sum, te_cnt = compute_target_avg_logprob(model, tokenizer, test_prompts, test_targets, batch_size=4)

tr_avg = tr_sum / (tr_cnt + 1e-8)
te_avg = te_sum / (te_cnt + 1e-8)

# -------------------------
# Multi-negative background stats (bucketed by job_role)
# -------------------------
def sample_bucketed_targets(rows: pd.DataFrame) -> List[str]:
    # For each row, sample a negative from the same bucket (job_role)
    negs = []
    for _, r in rows.iterrows():
        k = str(r.get(bucket_key, "UNK"))
        pool = buckets.get(k) or df["target"].tolist()
        cand = random.choice(pool)
        if cand == r["target"]:
            cand = random.choice(pool)
        negs.append(cand)
    return negs

def background_stats(prompts, rows_df, K=K_NEG, batch_size=4):
    # Return (mu, sd) for SUM log-prob, we’ll normalize by true token count later
    sums = []
    for _ in range(K):
        neg_targets = sample_bucketed_targets(rows_df)
        s, c = compute_target_avg_logprob(model, tokenizer, prompts, neg_targets, batch_size=batch_size)
        # Note: returns (sum, count); we accumulate sums to compute mean & std of sums
        sums.append(s)
    M = np.vstack(sums)           # (K, N)
    mu = M.mean(axis=0)
    sd = M.std(axis=0) + 1e-6
    return mu, sd

print(f"Computing background stats with K={K_NEG} (train)...")
tr_bg_mu_sum, tr_bg_sd_sum = background_stats(train_prompts, train_df, K=K_NEG)
print("Computing background stats (test)...")
te_bg_mu_sum, te_bg_sd_sum = background_stats(test_prompts, test_df, K=K_NEG)

# Normalize background by true token counts (to get per-token averages)
tr_bg_mu = tr_bg_mu_sum / (tr_cnt + 1e-8)
te_bg_mu = te_bg_mu_sum / (te_cnt + 1e-8)
tr_bg_sd = tr_bg_sd_sum / (tr_cnt + 1e-8)
te_bg_sd = te_bg_sd_sum / (te_cnt + 1e-8)

# -------------------------
# Z-score statistic (LiRA-style)
# -------------------------
tr_z = (tr_avg - tr_bg_mu) / tr_bg_sd
te_z = (te_avg - te_bg_mu) / te_bg_sd

y_true = np.concatenate([np.ones_like(tr_z), np.zeros_like(te_z)])
scores = np.concatenate([tr_z, te_z])
auc = roc_auc_score(y_true, scores)
print(f"Exposure (DistilGPT2) Z-score AUC: {auc:.4f}")

# -------------------------
# Save results & plots
# -------------------------
out = pd.DataFrame({
    "prompt": train_prompts + test_prompts,
    "target": train_targets + test_targets,
    "is_member": np.concatenate([np.ones(len(tr_z)), np.zeros(len(te_z))]).astype(int),
    "avg_logp_true": np.concatenate([tr_avg, te_avg]),
    "bg_mu_avg": np.concatenate([tr_bg_mu, te_bg_mu]),
    "bg_sd_avg": np.concatenate([tr_bg_sd, te_bg_sd]),
    "z_score": scores
})
out_csv = "exposure_results_gpt2.csv"
out.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(tr_avg, bins=20, alpha=0.7, label="train avg ll")
plt.hist(te_avg, bins=20, alpha=0.7, label="test avg ll")
plt.xlabel("Per-token avg log-prob (true target)")
plt.legend()
plt.title("True per-token averages")

plt.subplot(1,2,2)
plt.hist(tr_z, bins=20, alpha=0.7, label="train z")
plt.hist(te_z, bins=20, alpha=0.7, label="test z")
plt.xlabel("Z-score (true vs bucketed background)")
plt.legend()
plt.title("Exposure statistic (higher ⇒ more likely member)")
plt.tight_layout()
plt.savefig("exposure_histograms_gpt2.png")
print("Saved: exposure_histograms_gpt2.png")

# ROC curve
fpr, tpr, _ = roc_curve(y_true, scores)
roc_auc = calc_auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC={roc_auc:.4f})")
plt.plot([0,1], [0,1], lw=1, linestyle="--", label="Random")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Exposure Attack ROC (DistilGPT2, Z-score)")
plt.legend(loc="lower right")
plt.savefig("exposure_roc_curve_gpt2.png")
print("Saved: exposure_roc_curve_gpt2.png")

print("Done. Key AUC =", auc)
