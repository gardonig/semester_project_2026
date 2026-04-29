"""
LLM Poset Builder
=================

Uses the Claude API to answer anatomical ordering questions for each pair
of structures, building a full poset matrix automatically.

For each pair (A, B) and each axis (vertical, mediolateral, anteroposterior),
Claude is asked: "Is structure A above/left-of/anterior-to structure B?"
Answer: 1 (yes), -1 (no), 0 (unsure/ambiguous)

The resulting JSON is saved in the same format as the clinician-derived poset.

Usage
-----
    python scripts/llm_poset_builder.py \
        --structures_from data/posets/merged_sessions/merged_consensus.json \
        --output data/posets/llm_sessions/llm_claude.json

    # Dry run (no API calls, fills with 0s):
    python scripts/llm_poset_builder.py --dry_run

Requirements
------------
    pip install anthropic
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_POSET = PROJECT_ROOT / "data" / "posets" / "merged_sessions" / "merged_consensus.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "posets" / "llm_sessions" / "llm_claude_strict.json"

# ---------------------------------------------------------------------------
# Axis prompts
# ---------------------------------------------------------------------------

AXIS_PROMPTS = {
    "vertical": (
        "In the vertical (superior-inferior) axis of the human body, "
        "is the {a} strictly superior (higher up) to the {b}?\n"
        "Answer with exactly YES or NO. "
        "Only answer YES if you are confident this is true in typical adult anatomy. "
        "If you are uncertain or there is any ambiguity, answer NO."
    ),
    "mediolateral": (
        "In the mediolateral (left-right) axis of the human body, "
        "is the {a} strictly to the left of the {b} (from the patient's perspective)?\n"
        "Answer with exactly YES or NO. "
        "Only answer YES if you are confident this is true in typical adult anatomy. "
        "If you are uncertain or there is any ambiguity, answer NO."
    ),
    "anteroposterior": (
        "In the anteroposterior (front-back) axis of the human body, "
        "is the {a} strictly anterior (more towards the front) than the {b}?\n"
        "Answer with exactly YES or NO. "
        "Only answer YES if you are confident this is true in typical adult anatomy. "
        "If you are uncertain or there is any ambiguity, answer NO."
    ),
}

SYSTEM_PROMPT = (
    "You are an expert anatomist. Answer questions about the relative spatial "
    "positions of anatomical structures in the human body. Be precise and concise. "
    "Base your answers on typical adult human anatomy. "
    "You must answer only YES or NO — never any other word. "
    "Only say YES when you are fully confident. When in doubt, say NO."
)

# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def query_claude(client: anthropic.Anthropic, prompt: str, dry_run: bool = False) -> int:
    """
    Returns 1 (yes/above), -1 (no/not above), or 0 (unsure).
    """
    if dry_run:
        return 0

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = message.content[0].text.strip().upper()

    if answer.startswith("YES"):
        return 1
    else:
        return -1  # NO or any unexpected response → treat as NO


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_llm_poset(
    structures: list[str],
    dry_run: bool = False,
    delay: float = 0.1,
) -> dict:
    """
    Queries Claude for every pair on every axis and builds the poset dict.
    """
    client = anthropic.Anthropic()
    n = len(structures)

    # Initialize matrices with 0 (unknown)
    def empty_matrix():
        return [[0] * n for _ in range(n)]

    matrices = {
        "matrix_vertical":       empty_matrix(),
        "matrix_mediolateral":   empty_matrix(),
        "matrix_anteroposterior": empty_matrix(),
    }

    total_pairs = n * (n - 1) * 3
    done = 0

    for axis, matrix_key in [
        ("vertical",        "matrix_vertical"),
        ("mediolateral",    "matrix_mediolateral"),
        ("anteroposterior", "matrix_anteroposterior"),
    ]:
        template = AXIS_PROMPTS[axis]
        matrix = matrices[matrix_key]

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = -1  # a structure is not above itself
                    continue

                a = structures[i].replace("_", " ")
                b = structures[j].replace("_", " ")
                prompt = template.format(a=a, b=b)

                result = query_claude(client, prompt, dry_run=dry_run)
                matrix[i][j] = result
                done += 1

                if done % 50 == 0 or done == total_pairs:
                    print(f"  Progress: {done}/{total_pairs} queries")

                if not dry_run:
                    time.sleep(delay)

    return {
        "structures": [{"name": s} for s in structures],
        "matrix_vertical":       matrices["matrix_vertical"],
        "matrix_mediolateral":   matrices["matrix_mediolateral"],
        "matrix_anteroposterior": matrices["matrix_anteroposterior"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structures_from", default=str(DEFAULT_POSET),
                   help="Load structure list from this poset JSON")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Save resulting LLM poset to this JSON file")
    p.add_argument("--dry_run", action="store_true",
                   help="Skip API calls, fill matrix with 0s (for testing)")
    p.add_argument("--delay", type=float, default=0.1,
                   help="Seconds to wait between API calls (default 0.1)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load structure names from existing poset
    with open(args.structures_from) as f:
        poset = json.load(f)
    structures = [s["name"] for s in poset["structures"]]

    print(f"Building LLM poset for {len(structures)} structures")
    print(f"Total API calls: {len(structures) * (len(structures)-1) * 3}")
    print(f"Dry run: {args.dry_run}\n")

    result = build_llm_poset(structures, dry_run=args.dry_run, delay=args.delay)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
