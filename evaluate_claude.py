#!/usr/bin/env python3
"""Evaluate Claude Opus on Building On task via Anthropic API."""

import argparse
import json
import csv
import re
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEST_FILE = "together_val.jsonl"
STANDARDS_FILE = "standards.jsonl"
FEW_SHOT_FILE = "few_shot_examples.jsonl"
MODEL = "claude-opus-4-6"
MAX_TOKENS = 200
TEMPERATURE = 0
NUM_WORKERS = 50
GRANULARITIES = ["standard", "cluster", "domain", "grade"]
CCSS_PATTERN = re.compile(r"\b[K\d]+\.\w+\.\w+(?:\.\w+)?\b")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_test_data(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            text = entry["text"].removeprefix("<s>").removesuffix("</s>").strip()
            marker = "Building On Standards: "
            idx = text.index(marker)
            problem_text = text[:idx].removeprefix("Problem: ").strip()
            gold_str = text[idx + len(marker):]
            gold_codes = [c.strip() for c in gold_str.split(",") if c.strip()]
            examples.append({"problem": problem_text, "gold": gold_codes})
    return examples


def load_standards(path: str) -> str:
    """Load standards.jsonl and format as a reference list."""
    standards = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("level") == "Standard":
                standards.append(f'{entry["id"]}: {entry["description"]}')
    return "\n".join(standards)


def load_few_shot(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            text = entry["text"].removeprefix("<s>").removesuffix("</s>").strip()
            marker = "Building On Standards: "
            idx = text.index(marker)
            problem_text = text[:idx].removeprefix("Problem: ").strip()
            codes = text[idx + len(marker):]
            examples.append({"problem": problem_text, "codes": codes})
    return examples


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a K-12 math curriculum expert. Given a math problem, predict which Common Core State Standards (CCSS) the problem builds on — that is, what prerequisite standards a student needs to already know before attempting this problem. These are NOT the standards the problem directly tests, but the foundational skills it assumes.

Output ONLY a comma-separated list of CCSS codes (e.g. 3.OA.A.1, 5.NF.A.2). No explanations."""


def build_prompt(problem: str, standards_text: str, few_shot_examples: list[dict] | None = None) -> str:
    parts = []
    parts.append(f"Here is the full list of CCSS standards for reference:\n\n{standards_text}")
    if few_shot_examples:
        parts.append("\nHere are some examples:\n")
        for ex in few_shot_examples:
            parts.append(f"Problem:\n{ex['problem']}\n\nBuilding On Standards: {ex['codes']}")
            parts.append("---")
    parts.append(f"Problem:\n{problem}\n\nBuilding On Standards:")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def truncate_code(code: str, level: str) -> str:
    parts = code.split(".")
    if level == "standard":
        return code
    elif level == "cluster":
        return ".".join(parts[:3]) if len(parts) >= 3 else code
    elif level == "domain":
        return ".".join(parts[:2]) if len(parts) >= 2 else code
    elif level == "grade":
        return parts[0] if parts else code
    return code


def compute_metrics(predictions: list[list[str]], golds: list[list[str]]) -> dict:
    results = {}
    for level in GRANULARITIES:
        precisions, recalls, f1s, ems = [], [], [], []
        for pred, gold in zip(predictions, golds):
            pred_set = {truncate_code(c, level) for c in pred}
            gold_set = {truncate_code(c, level) for c in gold}
            if not pred_set and not gold_set:
                p, r, f1, em = 1.0, 1.0, 1.0, 1.0
            elif not pred_set or not gold_set:
                p, r, f1 = 0.0, 0.0, 0.0
                em = 1.0 if pred_set == gold_set else 0.0
            else:
                tp = len(pred_set & gold_set)
                p = tp / len(pred_set)
                r = tp / len(gold_set)
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                em = 1.0 if pred_set == gold_set else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            ems.append(em)
        n = len(golds)
        results[level] = {
            "precision": sum(precisions) / n,
            "recall": sum(recalls) / n,
            "f1": sum(f1s) / n,
            "exact_match": sum(ems) / n,
        }
    return results


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------
def call_claude(client: Anthropic, prompt: str) -> str:
    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            if attempt == 4:
                print(f"  FAILED after 5 attempts: {e}")
                return ""
            wait = 2 ** attempt
            if "rate" in str(e).lower() or "429" in str(e):
                wait = max(wait, 10)
            time.sleep(wait)
    return ""


def parse_codes(text: str) -> list[str]:
    text = text.split("\n")[0].strip()
    return CCSS_PATTERN.findall(text)


def run_inference(client: Anthropic, examples: list[dict], standards_text: str,
                  few_shot: list[dict] | None, num_workers: int) -> list[list[str]]:
    predictions = [None] * len(examples)

    def worker(i, ex):
        prompt = build_prompt(ex["problem"], standards_text, few_shot)
        raw = call_claude(client, prompt)
        return i, parse_codes(raw)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(worker, i, ex): i for i, ex in enumerate(examples)}
        done = 0
        for fut in as_completed(futures):
            i, codes = fut.result()
            predictions[i] = codes
            done += 1
            if done % 50 == 0:
                print(f"    {done}/{len(examples)} complete")

    return predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_eval(client, examples, standards_text, few_shot, label, num_workers, all_results):
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"{'='*60}")

    t0 = time.time()
    predictions = run_inference(client, examples, standards_text, few_shot, num_workers)
    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.1f}s")

    golds = [ex["gold"] for ex in examples]
    metrics = compute_metrics(predictions, golds)

    result = {"model": f"claude-opus-4-6-{label}", "step": "final", "epoch": "final"}
    for level in GRANULARITIES:
        for metric_name, val in metrics[level].items():
            result[f"{level}_{metric_name}"] = round(val, 6)
    all_results.append(result)

    # Print summary
    for level in GRANULARITIES:
        m = metrics[level]
        print(f"  {level:10s} F1: {m['f1']:.4f}  P: {m['precision']:.4f}  R: {m['recall']:.4f}  EM: {m['exact_match']:.4f}")

    # Save incrementally
    json_path = "results_claude_opus.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    csv_path = "all_results.csv"
    fieldnames = list(result.keys())
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
    print(f"  Saved to {json_path} and {csv_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Claude Opus on Building On task")
    parser.add_argument("--test-file", default=TEST_FILE)
    parser.add_argument("--standards-file", default=STANDARDS_FILE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--few-shot", action="store_true", help="Also run few-shot evaluation")
    parser.add_argument("--few-shot-only", action="store_true", help="Only run few-shot evaluation")
    parser.add_argument("--zero-shot-only", action="store_true", help="Only run zero-shot evaluation")
    args = parser.parse_args()

    client = Anthropic()
    examples = load_test_data(args.test_file)
    print(f"Loaded {len(examples)} test examples")

    standards_text = load_standards(args.standards_file)
    print(f"Loaded standards reference ({len(standards_text)} chars)")

    all_results = []

    # Zero-shot
    if not args.few_shot_only:
        run_eval(client, examples, standards_text, None, "zero-shot", args.workers, all_results)

    # Few-shot
    if args.few_shot or args.few_shot_only:
        if not os.path.exists(FEW_SHOT_FILE):
            print(f"\nWARNING: {FEW_SHOT_FILE} not found. Skipping few-shot eval.")
        else:
            few_shot = load_few_shot(FEW_SHOT_FILE)
            print(f"Loaded {len(few_shot)} few-shot examples")
            run_eval(client, examples, standards_text, few_shot, "few-shot", args.workers, all_results)

    print(f"\nDone. {len(all_results)} evaluations completed.")


if __name__ == "__main__":
    main()
