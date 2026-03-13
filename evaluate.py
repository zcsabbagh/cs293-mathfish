#!/usr/bin/env python3
"""Evaluate fine-tuned Qwen3 base models on Building On task via Together AI completions API."""

import argparse
import json
import csv
import re
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from together import Together

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEST_FILE = "together_val.jsonl"
MAX_TOKENS = 100
TEMPERATURE = 0
NUM_WORKERS = 20
GRANULARITIES = ["standard", "cluster", "domain", "grade"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_test_data(path: str) -> list[dict]:
    """Load test JSONL and split each entry into prompt + gold labels."""
    examples = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            text = entry["text"]
            # Strip <s> and </s> tokens
            text = text.removeprefix("<s>").removesuffix("</s>").strip()
            # Split at "Building On Standards: "
            marker = "Building On Standards: "
            idx = text.index(marker)
            prompt = "<s>" + text[: idx + len(marker)]
            gold_str = text[idx + len(marker) :]
            gold_codes = [c.strip() for c in gold_str.split(",") if c.strip()]
            examples.append({"prompt": prompt, "gold": gold_codes})
    return examples


# ---------------------------------------------------------------------------
# CCSS code truncation
# ---------------------------------------------------------------------------
def truncate_code(code: str, level: str) -> str | None:
    """Truncate a CCSS code to the given granularity level.

    Standard: 3.OA.A.1  -> 3.OA.A.1
    Cluster:  3.OA.A.1  -> 3.OA.A
    Domain:   3.OA.A.1  -> 3.OA
    Grade:    3.OA.A.1  -> 3
    """
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


def truncate_set(codes: list[str], level: str) -> set[str]:
    return {truncate_code(c, level) for c in codes}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(predictions: list[list[str]], golds: list[list[str]]) -> dict:
    """Compute per-problem P/R/F1 and exact match at all granularity levels."""
    results = {}
    for level in GRANULARITIES:
        precisions, recalls, f1s, ems = [], [], [], []
        for pred, gold in zip(predictions, golds):
            pred_set = truncate_set(pred, level)
            gold_set = truncate_set(gold, level)
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
# Parsing model output
# ---------------------------------------------------------------------------
CCSS_PATTERN = re.compile(r"\b[K\d]+\.\w+\.\w+(?:\.\w+)?\b")


def parse_codes(text: str) -> list[str]:
    """Extract CCSS codes from model output."""
    # Take only the first line / up to </s> or newline to avoid hallucinated extras
    text = text.split("</s>")[0].split("\n")[0].strip()
    return CCSS_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------
def call_completion(client: Together, model: str, prompt: str) -> str:
    """Single completion call with retry."""
    for attempt in range(3):
        try:
            resp = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stop=["</s>", "\n\n"],
            )
            return resp.choices[0].text
        except Exception as e:
            if attempt == 2:
                print(f"  FAILED after 3 attempts: {e}")
                return ""
            time.sleep(2 ** attempt)
    return ""


def run_inference(client: Together, model: str, examples: list[dict], num_workers: int = NUM_WORKERS) -> list[list[str]]:
    """Run inference on all examples in parallel."""
    predictions = [None] * len(examples)

    def worker(i, ex):
        raw = call_completion(client, model, ex["prompt"])
        return i, parse_codes(raw)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(worker, i, ex): i for i, ex in enumerate(examples)}
        done = 0
        for fut in as_completed(futures):
            i, codes = fut.result()
            predictions[i] = codes
            done += 1
            if done % 100 == 0:
                print(f"    {done}/{len(examples)} complete")

    return predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models on Building On task")
    parser.add_argument("model", help="Together model endpoint (e.g. account/model-name)")
    parser.add_argument("steps", nargs="+", type=int, help="Checkpoint step numbers")
    parser.add_argument("--epochs-per-step", type=float, default=None,
                        help="If set, epoch = step * epochs_per_step. Otherwise epoch = step index + 1.")
    parser.add_argument("--test-file", default=TEST_FILE, help="Path to test JSONL")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--include-final", action="store_true",
                        help="Also evaluate the final merged model (no step suffix)")
    args = parser.parse_args()

    num_workers = args.workers

    client = Together()
    examples = load_test_data(args.test_file)
    print(f"Loaded {len(examples)} test examples")

    # Build list of (model_string, step, epoch) to evaluate
    eval_runs = []
    sorted_steps = sorted(args.steps)
    for i, step in enumerate(sorted_steps):
        epoch = step * args.epochs_per_step if args.epochs_per_step else i + 1
        eval_runs.append((f"{args.model}:{step}", step, epoch))
    if args.include_final:
        max_epoch = eval_runs[-1][2] if eval_runs else 0
        step_size = (eval_runs[-1][2] - eval_runs[-2][2]) if len(eval_runs) >= 2 else 1
        eval_runs.append((args.model, "final", max_epoch + step_size))

    checkpoint_results = []
    for idx, (model_name, step, epoch) in enumerate(eval_runs):
        print(f"\n[{idx+1}/{len(eval_runs)}] Evaluating {model_name} (epoch {epoch})")

        t0 = time.time()
        predictions = run_inference(client, model_name, examples, num_workers)
        elapsed = time.time() - t0
        print(f"  Inference done in {elapsed:.1f}s")

        golds = [ex["gold"] for ex in examples]
        metrics = compute_metrics(predictions, golds)

        result = {"model": args.model, "step": step, "epoch": epoch}
        for level in GRANULARITIES:
            for metric_name, val in metrics[level].items():
                result[f"{level}_{metric_name}"] = round(val, 6)
        checkpoint_results.append(result)

        # Print summary
        sf1 = metrics["standard"]["f1"]
        cf1 = metrics["cluster"]["f1"]
        df1 = metrics["domain"]["f1"]
        gf1 = metrics["grade"]["f1"]
        print(f"  F1 — standard: {sf1:.4f}  cluster: {cf1:.4f}  domain: {df1:.4f}  grade: {gf1:.4f}")

        # Incremental save after each checkpoint
        safe_name = args.model.replace("/", "_").replace(":", "_")
        json_path = f"results_{safe_name}.json"
        with open(json_path, "w") as f:
            json.dump(checkpoint_results, f, indent=2)

        csv_path = "all_results.csv"
        fieldnames = list(result.keys())
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)
        print(f"  Saved to {json_path} and {csv_path}")

    print(f"\nDone. {len(checkpoint_results)} checkpoints evaluated.")


if __name__ == "__main__":
    main()
