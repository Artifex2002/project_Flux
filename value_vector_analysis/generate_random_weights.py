"""
Generate random tokens from SmolVLA's token space
==================================================

1. Loads SmolVLA policy to get the tokenizer and layer structure.
2. For each text-model layer, outputs the same number of "vectors" as reading_weights.py.
3. Each "vector" is top_k random token IDs decoded to strings; probability is set to 1/vocab_size (uniform placeholder).
4. Writes to a file formatted the same way as reading_weights_top_tokens.txt (prob= instead of logit=).
"""
import random
from pathlib import Path

import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

MODEL_ID = "HuggingFaceVLA/smolvla_libero"


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    device = select_device()
    print(f"Device: {device}")

    print(f"\nLoading policy: {MODEL_ID} ...")
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID).to(device).eval()

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    vocab_size = len(tokenizer)
    uniform_prob = 1.0 / vocab_size
    print(f"\nTokenizer vocab size: {vocab_size}")

    # Match reading_weights: same layer list (text_model down_proj only)
    down_proj_mats = []
    for name, module in policy.named_modules():
        if name.endswith("mlp.down_proj"):
            W = module.weight.detach()
            down_proj_mats.append((name, W))
    text_down_projs = [(n, w) for n, w in down_proj_mats if "text_model" in n]
    if not text_down_projs:
        print("No text_model down_proj layers found.")
        return

    num_vectors_per_layer = 10
    top_k = 30
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    out_path = Path(f"random_top_tokens(seed{seed}).txt")

    print(f"\nSampling {num_vectors_per_layer} random token vectors per layer, {top_k} tokens each.")
    print(f"Writing to {out_path} ...")

    lines = []
    lines.append(f"SmolVLA random tokens ({top_k} per vector, {num_vectors_per_layer} vectors per layer)")
    lines.append(f"Model: {MODEL_ID}  |  seed={seed}")
    lines.append("=" * 80)

    for name, W in text_down_projs:
        lines.append("")
        lines.append(f"Layer: {name}  (W.shape={W.shape})")
        lines.append("-" * 60)
        for vec_idx in range(num_vectors_per_layer):
            # Sample random token IDs from vocab (with replacement so we get top_k tokens)
            token_ids = [random.randint(0, vocab_size - 1) for _ in range(top_k)]
            tokens_out = [
                (tokenizer.decode([tid], skip_special_tokens=False), uniform_prob)
                for tid in token_ids
            ]
            lines.append(f"  Vector {vec_idx + 1}/{num_vectors_per_layer} (random):")
            for i, (token_str, prob) in enumerate(tokens_out):
                lines.append(f"    {i + 1:2d}. {repr(token_str):45s}  prob={prob:.6f}")
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Done. Wrote {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    main()
