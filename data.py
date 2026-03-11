"""Dataset download and loading for nanostate.

4 datasets:
  lm     - TinyShakespeare, byte-level language modeling
  lm-tok - FineWebEdu, BPE token-level language modeling (GPT-2 tokenizer)
  dna    - Nucleotide Transformer downstream tasks, sequence classification
  ts     - ETT (Electricity Transformer Temperature), multivariate forecasting

This file is fixed infrastructure. Iterate on train.py, not this.
"""

import os
import urllib.request

import numpy as np

DATA_DIR = "data"

# ---------------------------------------------------------------------------
# TinyShakespeare (byte-level language modeling)
# ---------------------------------------------------------------------------

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_shakespeare():
    os.makedirs(DATA_DIR, exist_ok=True)
    raw = os.path.join(DATA_DIR, "shakespeare.txt")
    if not os.path.exists(raw):
        print("Downloading TinyShakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, raw)
    with open(raw, "r") as f:
        text = f.read()
    data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    n = len(data)
    split = int(n * 0.9)
    np.save(os.path.join(DATA_DIR, "shakespeare_train.npy"), data[:split])
    np.save(os.path.join(DATA_DIR, "shakespeare_val.npy"), data[split:])
    print(f"Shakespeare: {split} train bytes, {n - split} val bytes")


def load_shakespeare(split):
    path = os.path.join(DATA_DIR, f"shakespeare_{split}.npy")
    if not os.path.exists(path):
        download_shakespeare()
    return np.load(path)


def get_batch_lm(data, batch_size, seq_len):
    """Random batch for next-byte prediction."""
    ix = np.random.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = np.stack([data[i : i + seq_len] for i in ix]).astype(np.int32)
    y = np.stack([data[i + 1 : i + seq_len + 1] for i in ix]).astype(np.int32)
    return x, y


# ---------------------------------------------------------------------------
# FineWebEdu (BPE token-level language modeling)
# ---------------------------------------------------------------------------

FINEWEB_DATASET = "HuggingFaceFW/FineWeb-Edu"
FINEWEB_SUBSET = "sample-10BT"
FINEWEB_TOKENS = 10_000_000  # ~10M tokens for train, ~1M for val


def download_fineweb():
    """Download FineWebEdu and tokenize with GPT-2 BPE."""
    cache = os.path.join(DATA_DIR, "fineweb")
    train_path = os.path.join(cache, "train.npy")
    val_path = os.path.join(cache, "val.npy")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return
    os.makedirs(cache, exist_ok=True)

    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    print(f"Downloading FineWebEdu ({FINEWEB_SUBSET})...")
    ds = load_dataset(FINEWEB_DATASET, FINEWEB_SUBSET, split="train", streaming=True)

    tokens = []
    target = FINEWEB_TOKENS + FINEWEB_TOKENS // 10  # train + val
    for row in ds:
        tokens.extend(enc.encode_ordinary(row["text"]))
        if len(tokens) >= target:
            break

    tokens = np.array(tokens[:target], dtype=np.int32)
    split = FINEWEB_TOKENS
    np.save(train_path, tokens[:split])
    np.save(val_path, tokens[split:])
    np.save(os.path.join(cache, "meta.npy"), np.array([enc.n_vocab]))
    print(f"FineWebEdu: {split:,} train tokens, {len(tokens) - split:,} val tokens, vocab={enc.n_vocab}")


def load_fineweb(split):
    path = os.path.join(DATA_DIR, "fineweb", f"{split}.npy")
    if not os.path.exists(path):
        download_fineweb()
    return np.load(path)


def get_fineweb_vocab_size():
    meta_path = os.path.join(DATA_DIR, "fineweb", "meta.npy")
    if not os.path.exists(meta_path):
        download_fineweb()
    return int(np.load(meta_path)[0])


# ---------------------------------------------------------------------------
# Nucleotide Transformer downstream tasks (DNA classification)
# ---------------------------------------------------------------------------

NUCLEOTIDE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def _encode_dna(seq):
    return np.array([NUCLEOTIDE_MAP.get(c, 4) for c in seq.upper()], dtype=np.int32)


def download_dna(task_name="promoter_no_tata"):
    """Download a Nucleotide Transformer downstream task from HuggingFace.

    Dataset is flat with a 'task' column. We filter by task_name.
    Available tasks: H2AFZ, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me2,
    H3K4me3, H3K9ac, H3K9me3, H4K20me1, enhancers, enhancers_types,
    promoter_all, promoter_no_tata, promoter_tata, splice_sites_acceptors,
    splice_sites_all, splice_sites_donors
    """
    cache = os.path.join(DATA_DIR, f"dna_{task_name}")
    if os.path.exists(cache):
        return
    os.makedirs(cache, exist_ok=True)
    print(f"Downloading DNA task: {task_name}...")
    from datasets import load_dataset

    full_ds = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")
    for split_name in full_ds:
        split_data = full_ds[split_name].filter(lambda row: row["task"] == task_name)
        seqs = [_encode_dna(row["sequence"]) for row in split_data]
        labels = np.array([row["label"] for row in split_data], dtype=np.int32)
        if len(seqs) == 0:
            continue
        # pad sequences to max length in split
        max_len = max(len(s) for s in seqs)
        padded = np.zeros((len(seqs), max_len), dtype=np.int32)
        lengths = np.zeros(len(seqs), dtype=np.int32)
        for i, s in enumerate(seqs):
            padded[i, : len(s)] = s
            lengths[i] = len(s)
        np.save(os.path.join(cache, f"{split_name}_seqs.npy"), padded)
        np.save(os.path.join(cache, f"{split_name}_labels.npy"), labels)
        np.save(os.path.join(cache, f"{split_name}_lengths.npy"), lengths)
    n_classes = len(set(labels))
    np.save(os.path.join(cache, "meta.npy"), np.array([n_classes, max_len]))
    print(f"DNA {task_name}: {n_classes} classes, max_len={max_len}, {len(labels)} samples")


def load_dna(split, task_name="promoter_no_tata"):
    cache = os.path.join(DATA_DIR, f"dna_{task_name}")
    if not os.path.exists(cache):
        download_dna(task_name)
    seqs = np.load(os.path.join(cache, f"{split}_seqs.npy"))
    labels = np.load(os.path.join(cache, f"{split}_labels.npy"))
    lengths = np.load(os.path.join(cache, f"{split}_lengths.npy"))
    meta = np.load(os.path.join(cache, "meta.npy"))
    return seqs, labels, lengths, int(meta[0]), int(meta[1])


def get_batch_dna(seqs, labels, batch_size):
    """Random batch for sequence classification."""
    ix = np.random.randint(0, len(seqs), (batch_size,))
    return seqs[ix], labels[ix]


# ---------------------------------------------------------------------------
# ETT - Electricity Transformer Temperature (time series forecasting)
# ---------------------------------------------------------------------------

ETT_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}


def download_ett(variant="ETTh1"):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{variant}.csv")
    if not os.path.exists(path):
        print(f"Downloading {variant}...")
        urllib.request.urlretrieve(ETT_URLS[variant], path)
    # parse CSV: skip date column, take 7 float columns
    import csv

    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            rows.append([float(x) for x in row[1:]])
    data = np.array(rows, dtype=np.float32)
    # normalize to zero mean, unit variance (per feature)
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8
    data = (data - mean) / std
    # standard split: 60/20/20 for ETTh, 60/20/20 for ETTm
    n = len(data)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    np.save(os.path.join(DATA_DIR, f"{variant}_train.npy"), data[:train_end])
    np.save(os.path.join(DATA_DIR, f"{variant}_val.npy"), data[train_end:val_end])
    np.save(os.path.join(DATA_DIR, f"{variant}_test.npy"), data[val_end:])
    np.save(os.path.join(DATA_DIR, f"{variant}_stats.npy"), np.stack([mean, std]))
    print(f"{variant}: {train_end} train, {val_end - train_end} val, {n - val_end} test rows, {data.shape[1]} features")


def load_ett(split, variant="ETTh1"):
    path = os.path.join(DATA_DIR, f"{variant}_{split}.npy")
    if not os.path.exists(path):
        download_ett(variant)
    return np.load(path)


def get_batch_ts(data, batch_size, seq_len, pred_len):
    """Random batch for time series forecasting.
    Returns (x, y) where x is the input window and y is the forecast target.
    x: (batch, seq_len, n_features)
    y: (batch, pred_len, n_features)
    """
    total_len = seq_len + pred_len
    ix = np.random.randint(0, len(data) - total_len, (batch_size,))
    x = np.stack([data[i : i + seq_len] for i in ix])
    y = np.stack([data[i + seq_len : i + total_len] for i in ix])
    return x, y


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def prepare_all():
    """Download and prep all datasets."""
    print("=== Preparing datasets ===")
    download_shakespeare()
    download_fineweb()
    download_dna()
    download_ett()
    print("=== Done ===")


if __name__ == "__main__":
    prepare_all()
