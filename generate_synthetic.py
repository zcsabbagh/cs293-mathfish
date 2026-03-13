#!/usr/bin/env python3
"""Generate synthetic math problems given Building On labels using Claude."""

import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic

TRAIN_FILE = "together_train.jsonl"
OUTPUT_FILE = "new_together_train.jsonl"
STANDARDS_FILE = "standards.jsonl"
MODEL = "claude-opus-4-6"
NUM_SYNTHETIC = 1200
NUM_WORKERS = 50

# Load standards descriptions for context
def load_standards(path: str) -> dict[str, str]:
    standards = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("level") == "Standard":
                standards[entry["id"]] = entry["description"]
    return standards


def load_train_data(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            text = entry["text"].removeprefix("<s>").removesuffix("</s>").strip()
            marker = "Building On Standards: "
            idx = text.index(marker)
            problem = text[:idx].removeprefix("Problem: ").strip()
            codes_str = text[idx + len(marker):]
            codes = [c.strip() for c in codes_str.split(",") if c.strip()]
            examples.append({
                "problem": problem,
                "codes": codes,
                "codes_str": codes_str,
                "raw": entry["text"],
            })
    return examples


SYSTEM_PROMPT = """You are a K-12 math curriculum expert who writes math problems. Given a set of prerequisite CCSS standards (Building On standards) and an example problem, generate a NEW math problem that requires the same prerequisite knowledge.

Rules:
- Use a COMPLETELY DIFFERENT real-world context or scenario (e.g. if the example is about sharing pizza, write about measuring ingredients, building fences, or saving money)
- Do NOT reuse the same activity structure, framing, or format as the example
- The new problem must require the same prerequisite standards to solve
- Match the difficulty level and grade appropriateness
- Output ONLY the problem text, nothing else. No labels, no explanation."""


def generate_problem(client: Anthropic, example: dict, standards: dict) -> str | None:
    standards_context = []
    for code in example["codes"]:
        desc = standards.get(code, "")
        if desc:
            standards_context.append(f"  {code}: {desc}")

    prompt = f"""Prerequisite standards (Building On):
{chr(10).join(standards_context) if standards_context else ', '.join(example['codes'])}

Example problem that uses these prerequisites:
{example['problem']}

Generate a new, different math problem that requires the same prerequisite knowledge:"""

    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=500,
                temperature=0.9,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            if attempt == 4:
                print(f"  FAILED after 5 attempts: {e}")
                return None
            wait = 2 ** attempt
            if "rate" in str(e).lower() or "429" in str(e):
                wait = max(wait, 10)
            time.sleep(wait)
    return None


def main():
    client = Anthropic()
    standards = load_standards(STANDARDS_FILE)
    print(f"Loaded {len(standards)} standard descriptions")

    train_data = load_train_data(TRAIN_FILE)
    print(f"Loaded {len(train_data)} training examples")

    # Sample 500 problems to augment
    random.seed(42)
    samples = random.sample(train_data, min(NUM_SYNTHETIC, len(train_data)))
    print(f"Generating {len(samples)} synthetic problems with {NUM_WORKERS} workers...")

    results = [None] * len(samples)

    def worker(i, ex):
        new_problem = generate_problem(client, ex, standards)
        return i, ex, new_problem

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(worker, i, ex): i for i, ex in enumerate(samples)}
        done = 0
        failed = 0
        for fut in as_completed(futures):
            i, ex, new_problem = fut.result()
            if new_problem:
                results[i] = {
                    "problem": new_problem,
                    "codes_str": ex["codes_str"],
                }
            else:
                failed += 1
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(samples)} complete ({failed} failed)")

    elapsed = time.time() - t0
    print(f"Generation done in {elapsed:.1f}s ({failed} failures)")

    # Write output: original + synthetic
    synthetic_count = 0
    with open(OUTPUT_FILE, "w") as out:
        # Write all original training data
        with open(TRAIN_FILE) as f:
            for line in f:
                out.write(line)

        # Write synthetic examples
        for r in results:
            if r is None:
                continue
            text = f"<s>Problem: {r['problem']}\n\nBuilding On Standards: {r['codes_str']}</s>"
            out.write(json.dumps({"text": text}) + "\n")
            synthetic_count += 1

    total = len(train_data) + synthetic_count
    print(f"\nWrote {OUTPUT_FILE}:")
    print(f"  Original: {len(train_data)}")
    print(f"  Synthetic: {synthetic_count}")
    print(f"  Total: {total}")


if __name__ == "__main__":
    main()
