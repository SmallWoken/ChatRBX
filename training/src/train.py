"""
Train TinyGPT on all .txt files in training/datasets/.

Usage (from training/):
    python src/train.py

Outputs:
    training/weights/checkpoint.pt   — model weights + vocab (reload to resume or export)
"""

import os
import sys
import time
import torch
from pathlib import Path

# Allow importing model.py from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from model import TinyGPT

# ---------------------------------------------------------------------------
# Config — tweak these freely
# ---------------------------------------------------------------------------
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
WEIGHTS_DIR  = Path(__file__).parent.parent / "weights"

# Model architecture
N_EMBD      = 32    # embedding dimension
N_HEAD      = 2     # attention heads (must divide N_EMBD)
N_LAYER     = 2     # transformer blocks
BLOCK_SIZE  = 64    # context window (characters)

# Training
MAX_ITERS   = 5000  # total gradient steps
BATCH_SIZE  = 32    # sequences per batch
LR          = 3e-3  # learning rate
EVAL_EVERY  = 500   # print loss every N iters
SAMPLE_EVERY = 1000 # print a generated sample every N iters
SAMPLE_LEN  = 200   # characters to generate in each sample

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------------


def load_corpus() -> str:
    txt_files = sorted(DATASETS_DIR.glob("*.txt"))
    if not txt_files:
        sys.exit(
            f"No .txt files found in {DATASETS_DIR}.\n"
            "Add some text files there and re-run train.py."
        )
    corpus = ""
    for f in txt_files:
        print(f"  Loading {f.name} ({f.stat().st_size // 1024} KB)")
        corpus += f.read_text(encoding="utf-8", errors="replace")
    print(f"Corpus: {len(corpus):,} characters from {len(txt_files)} file(s)\n")
    return corpus


def build_vocab(corpus: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    chars = sorted(set(corpus))
    stoi  = {ch: i for i, ch in enumerate(chars)}
    itos  = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[ch] for ch in text if ch in stoi]


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i     : i + block_size    ] for i in ix])
    y  = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
    n_batches: int = 20,
) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, block_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def sample(model: TinyGPT, itos: dict[int, str], device: str, length: int = 200) -> str:
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=length, temperature=0.8, top_k=40)
    model.train()
    return "".join(itos[i] for i in out[0].tolist())


def main():
    print("=== chatrbx TinyGPT trainer ===\n")
    print(f"Device: {DEVICE}")

    # Load data
    print(f"\nLoading datasets from {DATASETS_DIR}:")
    corpus = load_corpus()

    # Vocab
    chars, stoi, itos = build_vocab(corpus)
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size} unique characters")

    # Encode full corpus
    data = torch.tensor(encode(corpus, stoi), dtype=torch.long)

    # Train/val split (90/10)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data   = data[n:]
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}\n")

    # Model
    model = TinyGPT(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        block_size=BLOCK_SIZE,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Training loop
    print(f"\nTraining for {MAX_ITERS} iterations...\n")
    t0 = time.time()

    for step in range(1, MAX_ITERS + 1):
        x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % EVAL_EVERY == 0 or step == 1:
            val_loss = estimate_loss(model, val_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            elapsed  = time.time() - t0
            print(f"step {step:>5}/{MAX_ITERS} | train loss {loss.item():.4f} | val loss {val_loss:.4f} | {elapsed:.1f}s")

        if step % SAMPLE_EVERY == 0:
            print("\n--- sample ---")
            print(sample(model, itos, DEVICE, SAMPLE_LEN))
            print("--- end sample ---\n")

    print(f"\nTraining complete in {time.time() - t0:.1f}s")

    # Save checkpoint
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "vocab": {
            "chars": chars,         # list of chars in index order
            "stoi": stoi,
            "itos": {str(k): v for k, v in itos.items()},  # JSON-safe keys
        },
        "config": {
            "vocab_size": vocab_size,
            "n_embd":     N_EMBD,
            "n_head":     N_HEAD,
            "n_layer":    N_LAYER,
            "block_size": BLOCK_SIZE,
        },
    }
    ckpt_path = WEIGHTS_DIR / "checkpoint.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint saved → {ckpt_path}")
    print("\nRun  python src/export.py  to generate Roblox-ready weight files.")


if __name__ == "__main__":
    main()
