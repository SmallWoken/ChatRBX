"""
Train TinyGPT on a tab-separated chatbot dataset.

Expected dataset format:
    training/datasets/chatbot dataset.txt

Each non-empty line should look like:
    prompt<TAB>reply

Usage (from training/):
    python src/train.py

Outputs:
    training/weights/checkpoint.pt
"""

import sys
import time
from pathlib import Path

import torch

# Allow importing model.py from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from model import TinyGPT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
DATA_FILE = DATASETS_DIR / "chatbot dataset.txt"

# Model architecture
N_EMBD = 96
N_HEAD = 4
N_LAYER = 4
BLOCK_SIZE = 96

# Training
MAX_ITERS = 15000
BATCH_SIZE = 32
LR = 8e-4
EVAL_EVERY = 500
SAMPLE_EVERY = 1000
SAMPLE_LEN = 120
GRAD_CLIP = 1.0
TRAIN_SPLIT = 0.9
SEED = 1337

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ").strip()
    text = " ".join(text.split())
    return text


def load_examples() -> list[str]:
    if not DATA_FILE.exists():
        sys.exit(f"Missing dataset file: {DATA_FILE}")

    raw = DATA_FILE.read_text(encoding="utf-8", errors="replace").splitlines()

    examples: list[str] = []
    seen: set[str] = set()
    skipped = 0

    for line in raw:
        line = line.strip()
        if not line:
            continue

        if "\t" not in line:
            skipped += 1
            continue

        user, bot = line.split("\t", 1)
        user = normalize_text(user)
        bot = normalize_text(bot)

        if not user or not bot:
            skipped += 1
            continue

        # Filter out absurdly long rows for this tiny char model
        if len(user) > 120 or len(bot) > 180:
            skipped += 1
            continue

        ex = f"<BOS>User: {user}\nBot: {bot}<EOS>"

        # Deduplicate exact repeats
        if ex in seen:
            continue
        seen.add(ex)
        examples.append(ex)

    if not examples:
        sys.exit("No valid examples found in dataset.")

    print(f"Loaded {len(examples)} cleaned examples from {DATA_FILE.name}")
    if skipped:
        print(f"Skipped {skipped} malformed/empty/too-long line(s)")

    print("\nExample:")
    print(examples[0])
    print()

    return examples


def build_vocab(examples: list[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    corpus = "\n\n".join(examples)
    chars = sorted(set(corpus))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[ch] for ch in text if ch in stoi]


def encode_examples(examples: list[str], stoi: dict[str, int]) -> list[torch.Tensor]:
    encoded: list[torch.Tensor] = []
    for ex in examples:
        ids = encode(ex, stoi)
        if len(ids) >= 2:
            encoded.append(torch.tensor(ids, dtype=torch.long))
    return encoded


def split_examples(examples: list[str], train_split: float) -> tuple[list[str], list[str]]:
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(examples), generator=g).tolist()
    shuffled = [examples[i] for i in perm]

    n_train = max(1, int(len(shuffled) * train_split))
    if n_train >= len(shuffled):
        n_train = len(shuffled) - 1

    train_examples = shuffled[:n_train]
    val_examples = shuffled[n_train:]

    if not val_examples:
        val_examples = train_examples[-1:]
        train_examples = train_examples[:-1]

    return train_examples, val_examples


def get_batch(
    encoded_examples: list[torch.Tensor],
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = [ex for ex in encoded_examples if len(ex) >= block_size + 1]

    if not valid:
        raise ValueError(
            f"No examples are at least {block_size + 1} chars long. "
            f"Lower BLOCK_SIZE."
        )

    x_list = []
    y_list = []

    for _ in range(batch_size):
        ex = valid[torch.randint(len(valid), (1,)).item()]
        start = torch.randint(0, len(ex) - block_size, (1,)).item()
        chunk = ex[start : start + block_size + 1]
        x_list.append(chunk[:-1])
        y_list.append(chunk[1:])

    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    encoded_examples: list[torch.Tensor],
    block_size: int,
    batch_size: int,
    device: str,
    n_batches: int = 20,
) -> float:
    model.eval()
    losses = []

    for _ in range(n_batches):
        x, y = get_batch(encoded_examples, batch_size, block_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def decode(ids: list[int], itos: dict[int, str]) -> str:
    return "".join(itos[i] for i in ids)


def clean_generated_reply(text: str) -> str:
    if "Bot:" in text:
        text = text.split("Bot:", 1)[1]
    if "<EOS>" in text:
        text = text.split("<EOS>", 1)[0]
    return text.strip()


@torch.no_grad()
def sample_reply(
    model: TinyGPT,
    stoi: dict[str, int],
    itos: dict[int, str],
    prompt: str,
    device: str,
    max_new_tokens: int = 120,
) -> str:
    seed = f"<BOS>User: {prompt}\nBot:"
    idx = torch.tensor([encode(seed, stoi)], dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=20,
    )

    text = decode(out[0].tolist(), itos)
    return clean_generated_reply(text)


def main():
    torch.manual_seed(SEED)

    print("=== chatrbx TinyGPT trainer ===\n")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATA_FILE}")

    examples = load_examples()
    train_examples, val_examples = split_examples(examples, TRAIN_SPLIT)

    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples:   {len(val_examples)}")

    chars, stoi, itos = build_vocab(examples)
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}")

    train_encoded = encode_examples(train_examples, stoi)
    val_encoded = encode_examples(val_examples, stoi)

    model = TinyGPT(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        block_size=BLOCK_SIZE,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"\nTraining for {MAX_ITERS} iterations...\n")
    t0 = time.time()

    sample_prompts = [
        "Hello",
        "What is AI?",
        "Do you like music?",
        "Tell me a joke",
        "How are you doing?",
    ]

    for step in range(1, MAX_ITERS + 1):
        x, y = get_batch(train_encoded, BATCH_SIZE, BLOCK_SIZE, DEVICE)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % EVAL_EVERY == 0 or step == 1:
            val_loss = estimate_loss(model, val_encoded, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            elapsed = time.time() - t0
            print(
                f"step {step:>5}/{MAX_ITERS} | "
                f"train loss {loss.item():.4f} | "
                f"val loss {val_loss:.4f} | "
                f"{elapsed:.1f}s"
            )

        if step % SAMPLE_EVERY == 0:
            print("\n--- samples ---")
            for prompt in sample_prompts:
                reply = sample_reply(model, stoi, itos, prompt, DEVICE, SAMPLE_LEN)
                print(f"User: {prompt}")
                print(f"Bot:  {reply}\n")
            print("--- end samples ---\n")

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.1f}s")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "vocab": {
            "chars": chars,
            "stoi": stoi,
            "itos": {str(k): v for k, v in itos.items()},
        },
        "config": {
            "vocab_size": vocab_size,
            "n_embd": N_EMBD,
            "n_head": N_HEAD,
            "n_layer": N_LAYER,
            "block_size": BLOCK_SIZE,
        },
        "meta": {
            "dataset_file": str(DATA_FILE),
            "num_examples": len(examples),
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
        },
    }

    ckpt_path = WEIGHTS_DIR / "checkpoint.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint saved -> {ckpt_path}")
    print("\nRun python src/export.py to generate Roblox-ready weight files.")


if __name__ == "__main__":
    main()