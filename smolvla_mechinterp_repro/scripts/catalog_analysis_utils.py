from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator


ANGLE_TOKEN_RE = re.compile(r"^<[^>]+>$")
ACTION_TOKEN_RE = re.compile(r"^<ac\d+>$", re.IGNORECASE)
WORDISH_TOKEN_RE = re.compile(r"^[a-z][a-z'\-]*$")


def iter_catalog_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_token_text(token: str) -> str:
    return token.strip().lower()


def alpha_only(token: str) -> str:
    return "".join(ch for ch in normalize_token_text(token) if ch.isalpha())


def is_angle_token(token: str) -> bool:
    return bool(ANGLE_TOKEN_RE.match(normalize_token_text(token)))


def is_action_token(token: str) -> bool:
    return bool(ACTION_TOKEN_RE.match(normalize_token_text(token)))


def is_wordish_token(token: str) -> bool:
    stripped = normalize_token_text(token)
    return bool(WORDISH_TOKEN_RE.match(stripped))


def simple_stem(token: str) -> str:
    word = alpha_only(token)
    if len(word) <= 3:
        return word
    if word.endswith("ingly") and len(word) > 7:
        return word[:-5]
    if word.endswith("edly") and len(word) > 6:
        return word[:-4]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("ly") and len(word) > 4:
        return word[:-2]
    if word.endswith("iest") and len(word) > 6:
        return word[:-4] + "y"
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"
    if word.endswith("est") and len(word) > 5:
        return word[:-3]
    if word.endswith("er") and len(word) > 4:
        return word[:-2]
    if word.endswith(("ches", "shes", "sses", "xes", "zes")) and len(word) > 5:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        return word[:-1]
    for suffix in ("ation", "ations", "ition", "ments", "ment", "ness", "less", "able", "ible", "ities", "ity", "ally"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def weighted_rank_score(ranks: list[int]) -> float:
    return sum(1.0 / (rank + 1) for rank in ranks)
