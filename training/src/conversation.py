"""
Train TinyGPT on paired conversational data from:

    training/datasets/input_texts.txt
    training/datasets/label_texts.txt

Each line in input_texts.txt must match the same line number in label_texts.txt.

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
# Config — tweak these freely
# ---------------------------------------------------------------------------
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
WEIGHTS_DIR = Path(__file__).parent.parent / "weights"

INPUT_FILE = DATASETS_DIR / "input_texts_optimized.txt"
LABEL_FILE = DATASETS_DIR / "label_texts_optimized.txt"

# Model architecture
N_EMBD = 64
N_HEAD = 4              # must divide N_EMBD
N_LAYER = 4
BLOCK_SIZE = 64       # bigger context helps for chat patterns

# Training
MAX_ITERS = 15000
BATCH_SIZE = 32
LR = 8e-4
EVAL_EVERY = 500
SAMPLE_EVERY = 2000
SAMPLE_LEN = 200
GRAD_CLIP = 1.0
TRAIN_SPLIT = 0.9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1337
# ---------------------------------------------------------------------------


def clean_special_tokens(text: str) -> str:
    """Remove existing [sos]/[eos] markers so we can format consistently."""
    text = text.replace("[sos]", "").replace("[eos]", "")
    return " ".join(text.strip().split())


def load_examples() -> list[str]:
    if not INPUT_FILE.exists():
        sys.exit(f"Missing input file: {INPUT_FILE}")
    if not LABEL_FILE.exists():
        sys.exit(f"Missing label file: {LABEL_FILE}")

    inputs = INPUT_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    labels = LABEL_FILE.read_text(encoding="utf-8", errors="replace").splitlines()

    if len(inputs) != len(labels):
        sys.exit(
            f"Line count mismatch:\n"
            f"  {INPUT_FILE.name}: {len(inputs)} lines\n"
            f"  {LABEL_FILE.name}: {len(labels)} lines"
        )

    examples = []
    skipped = 0

    for i, (inp, out) in enumerate(zip(inputs, labels), start=1):
        inp = clean_special_tokens(inp)
        out = clean_special_tokens(out)

        if not inp or not out:
            skipped += 1
            continue

        example = f"<BOS>User: {inp}\nBot: {out}<EOS>"
        examples.append(example)

    if not examples:
        sys.exit("No valid training examples found after cleaning.")

    print(f"Loaded {len(examples)} paired examples")
    if skipped:
        print(f"Skipped {skipped} empty example(s)")

    print("\nExample training sample:\n")
    print(examples[0][:300] + ("..." if len(examples[0]) > 300 else ""))
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
    encoded = []
    for ex in examples:
        ids = encode(ex, stoi)
        if len(ids) >= 2:
            encoded.append(torch.tensor(ids, dtype=torch.long))
    return encoded


def split_examples(examples: list[str], train_split: float = 0.9) -> tuple[list[str], list[str]]:
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


def get_batch(encoded_examples, batch_size, block_size, device):
    valid = [ex for ex in encoded_examples if len(ex) >= block_size + 1]

    if not valid:
        raise ValueError(
            f"No examples are at least {block_size + 1} tokens long. "
            f"Lower BLOCK_SIZE or use longer training examples."
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


def extract_bot_reply(generated_text: str) -> str:
    marker = "Bot:"
    if marker in generated_text:
        text = generated_text.split(marker, 1)[1]
    else:
        text = generated_text

    if "<EOS>" in text:
        text = text.split("<EOS>", 1)[0]

    return text.strip()


@torch.no_grad()
def sample_reply(
    model: TinyGPT,
    stoi: dict[str, int],
    itos: dict[int, str],
    device: str,
    prompt: str,
    max_new_tokens: int = 160,
) -> str:
    prompt = clean_special_tokens(prompt)
    seed_text = f"<BOS>User: {prompt}\nBot:"
    idx = torch.tensor([encode(seed_text, stoi)], dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=40,
    )

    generated = decode(out[0].tolist(), itos)
    return extract_bot_reply(generated)


def main():
    torch.manual_seed(SEED)

    print("=== chatrbx TinyGPT trainer ===\n")
    print(f"Device: {DEVICE}")

    print(f"\nLoading paired datasets from {DATASETS_DIR}:")
    examples = load_examples()

    train_examples, val_examples = split_examples(examples, TRAIN_SPLIT)
    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples:   {len(val_examples)}\n")

    chars, stoi, itos = build_vocab(examples)
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size} unique characters")

    train_encoded = encode_examples(train_examples, stoi)
    val_encoded = encode_examples(val_examples, stoi)

    if not train_encoded or not val_encoded:
        sys.exit("Training or validation set ended up empty after encoding.")

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
        "hi, how are you doing?",
        "what are you doing this weekend?",
        "do you like music?",
    ]

    for step in range(1, MAX_ITERS + 1):
        x, y = get_batch(train_encoded, BATCH_SIZE, BLOCK_SIZE, DEVICE)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % EVAL_EVERY == 0 or step == 1:
            train_loss = loss.item()
            val_loss = estimate_loss(model, val_encoded, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            elapsed = time.time() - t0
            print(
                f"step {step:>5}/{MAX_ITERS} | "
                f"train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} | "
                f"{elapsed:.1f}s"
            )

        if step % SAMPLE_EVERY == 0:
            print("\n--- samples ---")
            for prompt in sample_prompts:
                reply = sample_reply(model, stoi, itos, DEVICE, prompt, SAMPLE_LEN)
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
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "input_file": str(INPUT_FILE),
            "label_file": str(LABEL_FILE),
        },
    }

    ckpt_path = WEIGHTS_DIR / "checkpoint.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint saved -> {ckpt_path}")
    print("\nRun python src/export.py to generate Roblox-ready weight files.")


if __name__ == "__main__":
    main()