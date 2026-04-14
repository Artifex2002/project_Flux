"""
Reading SmolVLA weights: value vector → readable token
======================================================

1. Loads SmolVLA policy and gets the decoding matrix (lm_head).
2. For each text-model layer, samples random value vectors (columns of down_proj).
3. Applies lm_head to get vocab logits, converts to softmax probabilities, and decodes to top tokens; writes results to a file.
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


def vector_to_token(
    vector: torch.Tensor,
    lm_head: torch.nn.Linear,
    tokenizer,
    top_k: int = 1,
) -> list[tuple[str, float]]:
    """
    Decode a hidden-state vector to token(s) using the VLM's lm_head.

    Args:
        vector: shape (hidden_size,) or (1, hidden_size); will be moved to lm_head device.
        lm_head: the decoding linear layer (vocab_size, hidden_size).
        tokenizer: tokenizer used by the model (e.g. policy.model.vlm_with_expert.processor.tokenizer).
        top_k: number of top tokens to return (by probability).

    Returns:
        List of (token_string, probability) for the top_k tokens.
    """
    device = next(lm_head.parameters()).device
    dtype = next(lm_head.parameters()).dtype
    v = vector.detach().to(device=device, dtype=dtype)
    if v.dim() == 1:
        v = v.unsqueeze(0)
    # lm_head: (vocab_size, hidden_size) -> logits (batch, vocab_size)
    logits = lm_head(v).squeeze(0)
    probs = torch.softmax(logits, dim=-1)
    scores, indices = probs.topk(top_k, dim=-1)
    out = []
    for i in range(indices.numel()):
        tid = indices[i].item()
        prob = scores[i].item()
        token_str = tokenizer.decode([tid], skip_special_tokens=False)
        out.append((token_str, prob))
    return out


def main() -> None:
    device = select_device()
    print(f"Device: {device}")

    print(f"\nLoading policy: {MODEL_ID} ...")
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID).to(device).eval()

    # Decoding matrix: hidden_size -> vocab_size (this turns a value vector into token logits)
    lm_head = policy.model.vlm_with_expert.vlm.lm_head
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer

    hidden_size = lm_head.weight.shape[1]
    vocab_size = lm_head.weight.shape[0]
    print(f"\nDecoding matrix (lm_head): weight shape (vocab_size={vocab_size}, hidden_size={hidden_size})")

    # Collect all down_proj layers; use only text-model ones so dimension matches lm_head
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
    out_path = Path(f"value_vectors_top_tokens(seed{seed}).txt")

    print(f"\nDecoding {num_vectors_per_layer} random value vectors per layer, top-{top_k} tokens each.")
    print(f"Writing to {out_path} ...")

    lines = []
    lines.append(f"SmolVLA top-{top_k} tokens per random value vector (10 vectors per layer)")
    lines.append(f"Model: {MODEL_ID}  |  seed={seed}")
    lines.append("=" * 80)

    for name, W in text_down_projs:
        _, num_cols = W.shape
        col_indices = random.sample(range(num_cols), min(num_vectors_per_layer, num_cols))
        lines.append("")
        lines.append(f"Layer: {name}  (W.shape={W.shape})")
        lines.append("-" * 60)
        for vec_idx, col_idx in enumerate(col_indices):
            single_column = W[:, col_idx].to(device)
            if single_column.shape[0] != hidden_size:
                single_column = lm_head.weight[0].detach().to(device)
            tokens_out = vector_to_token(single_column, lm_head, tokenizer, top_k=top_k)
            lines.append(f"  Vector {vec_idx + 1}/{len(col_indices)} (column {col_idx}):")
            for i, (token_str, prob) in enumerate(tokens_out):
                lines.append(f"    {i + 1:2d}. {repr(token_str):45s}  prob={prob:.6f}")
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Done. Wrote {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    main()