#!/usr/bin/env python3
"""
Generate 1-sentence example math problems for each standard in multi-option clusters (3+).
Caches results to experiments/standard_examples.json.
"""

import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from together import Together

load_dotenv()
client = Together()

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
STANDARDS_PATH = os.path.join(os.path.dirname(__file__), "..", "standards.jsonl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "standard_examples.json")


def load_standards(path):
    standards = {}
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            standards[e["id"]] = e
    return standards


def get_multi_option_clusters(standards):
    """Return clusters with 3+ standards, and their standard IDs."""
    stds_by_cluster = defaultdict(dict)
    for sid, entry in standards.items():
        if entry["level"] == "Standard":
            parent = entry.get("parent", "")
            stds_by_cluster[parent][sid] = entry["description"]
    return {k: v for k, v in stds_by_cluster.items() if len(v) >= 3}


def generate_example(std_id, description, max_retries=5):
    prompt = f"For the Common Core standard {std_id}: {description}, write ONE brief example math problem (1 sentence) that a student would solve."
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(2 ** attempt)
            else:
                raise
    return ""


def main():
    standards = load_standards(STANDARDS_PATH)
    multi_clusters = get_multi_option_clusters(standards)

    # Collect all standard IDs we need examples for
    needed = {}
    for cluster_id, stds in multi_clusters.items():
        for sid, desc in stds.items():
            needed[sid] = desc

    print(f"Generating examples for {len(needed)} standards in {len(multi_clusters)} clusters (3+ standards)")

    # Load existing cache
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached examples")
    else:
        cache = {}

    # Filter to only what we need
    todo = {sid: desc for sid, desc in needed.items() if sid not in cache}
    print(f"Need to generate {len(todo)} new examples")

    if not todo:
        print("All examples already cached.")
        return

    done = 0
    def worker(sid_desc):
        sid, desc = sid_desc
        return sid, generate_example(sid, desc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(worker, (sid, desc)): sid for sid, desc in todo.items()}
        for future in as_completed(futures):
            sid, example = future.result()
            cache[sid] = example
            done += 1
            if done % 20 == 0:
                print(f"  [{done}/{len(todo)}] Generated example for {sid}")
                # Save incrementally
                with open(OUTPUT_PATH, "w") as f:
                    json.dump(cache, f, indent=2)

    # Final save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Saved {len(cache)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
