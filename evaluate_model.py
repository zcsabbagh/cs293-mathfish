#!/usr/bin/env python3
"""Evaluate a fine-tuned model on Building On task via Together AI completions API.

Usage:
  python evaluate_model.py MODEL_NAME [--label LABEL] [--workers N]

Examples:
  python evaluate_model.py zcs2024_8faa/Qwen3-14B-synth-abc123
  python evaluate_model.py zcs2024_8faa/Qwen3-14B-synth-abc123 --label "Qwen3-14B+synth"
"""

import argparse
import json
import csv
import re
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from together import Together

TEST_FILE = "together_val.jsonl"
MAX_TOKENS = 100
TEMPERATURE = 0
NUM_WORKERS = 50
GRANULARITIES = ["standard", "cluster", "domain", "grade"]
CCSS_PATTERN = re.compile(r"\b[K\d]+\.\w+\.\w+(?:\.\w+)?\b")


def load_test_data(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            text = entry["text"].removeprefix("<s>").removesuffix("</s>").strip()
            marker = "Building On Standards: "
            idx = text.index(marker)
            prompt = "<s>" + text[:idx + len(marker)]
            gold_str = text[idx + len(marker):]
            gold_codes = [c.strip() for c in gold_str.split(",") if c.strip()]
            examples.append({"prompt": prompt, "gold": gold_codes})
    return examples


def truncate_code(code: str, level: str) -> str:
    parts = code.split(".")
    if level == "standard": return code
    elif level == "cluster": return ".".join(parts[:3]) if len(parts) >= 3 else code
    elif level == "domain": return ".".join(parts[:2]) if len(parts) >= 2 else code
    elif level == "grade": return parts[0] if parts else code
    return code


def compute_metrics(predictions, golds):
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
            precisions.append(p); recalls.append(r); f1s.append(f1); ems.append(em)
        n = len(golds)
        results[level] = {
            "precision": sum(precisions) / n,
            "recall": sum(recalls) / n,
            "f1": sum(f1s) / n,
            "exact_match": sum(ems) / n,
        }
    return results


def parse_codes(text: str) -> list[str]:
    text = text.split("</s>")[0].split("\n")[0].strip()
    return CCSS_PATTERN.findall(text)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on Building On task")
    parser.add_argument("model", help="Together model endpoint")
    parser.add_argument("--label", default=None, help="Label for results (default: model name)")
    parser.add_argument("--test-file", default=TEST_FILE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    label = args.label or args.model
    client = Together()
    examples = load_test_data(args.test_file)
    print(f"Loaded {len(examples)} test examples")

    predictions = [None] * len(examples)

    def worker(i, ex):
        for attempt in range(10):
            try:
                resp = client.completions.create(
                    model=args.model, prompt=ex["prompt"],
                    max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                    stop=["</s>", "\n\n"],
                )
                return i, parse_codes(resp.choices[0].text)
            except Exception as e:
                if attempt == 9:
                    print(f"  FAILED after 10 attempts: {e}")
                    return i, []
                wait = min(2 ** attempt, 30)
                if "not running" in str(e).lower() or "endpoint" in str(e).lower():
                    wait = max(wait, 15)
                time.sleep(wait)
        return i, []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, i, ex): i for i, ex in enumerate(examples)}
        done = 0
        for fut in as_completed(futures):
            i, codes = fut.result()
            predictions[i] = codes
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(examples)} complete")

    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s")

    golds = [ex["gold"] for ex in examples]
    metrics = compute_metrics(predictions, golds)

    # Print results
    print(f"\nResults for {label}:")
    for level in GRANULARITIES:
        m = metrics[level]
        print(f"  {level:10s} F1: {m['f1']:.4f}  P: {m['precision']:.4f}  R: {m['recall']:.4f}  EM: {m['exact_match']:.4f}")

    # Save JSON
    safe = label.replace(" ", "_").replace("/", "_").replace("+", "_")
    result = {"model": label, "step": "final", "epoch": "final"}
    for level in GRANULARITIES:
        for metric_name, val in metrics[level].items():
            result[f"{level}_{metric_name}"] = round(val, 6)

    json_path = f"results_{safe}.json"
    with open(json_path, "w") as f:
        json.dump([result], f, indent=2)

    csv_path = "all_results.csv"
    fieldnames = list(result.keys())
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    print(f"Saved to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
