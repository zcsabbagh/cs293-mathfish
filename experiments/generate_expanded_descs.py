#!/usr/bin/env python3
"""
Generate expanded, discriminative descriptions for standards in multi-option clusters.
Uses Together AI to create 1-sentence descriptions focused on what makes each standard
DIFFERENT from others in the same cluster.

Saves output to experiments/expanded_standards.json
"""

import json
import os
import sys
import time
from collections import defaultdict
from dotenv import load_dotenv
from together import Together

load_dotenv()
client = Together()

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
STANDARDS_PATH = os.path.join(os.path.dirname(__file__), "..", "standards.jsonl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "expanded_standards.json")


def load_standards(path):
    standards = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            standards[entry["id"]] = entry
    return standards


def get_clusters_with_standards(standards):
    """Return {cluster_id: {std_id: description}} for clusters with 2+ standards."""
    standards_by_cluster = defaultdict(dict)
    for sid, entry in standards.items():
        if entry["level"] == "Standard":
            parent = entry.get("parent", "")
            standards_by_cluster[parent][sid] = entry["description"]
    # Filter to clusters with 2+ standards
    return {k: v for k, v in standards_by_cluster.items() if len(v) >= 2}


def generate_descriptions(cluster_id, stds_dict, max_retries=5):
    """Ask the model to generate discriminative descriptions for standards in a cluster."""
    std_list = "\n".join(f"- {sid}: {desc}" for sid, desc in sorted(stds_dict.items()))

    prompt = f"""These Common Core math standards are in cluster {cluster_id} and are commonly confused with each other. For each one, write a 1-sentence description (max 25 words) focused on what makes it DIFFERENT from the others. Focus on the key distinguishing verb or concept.

Standards:
{std_list}

Respond in this exact format, one per line:
STANDARD_ID: description
"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            return parse_response(text, stds_dict)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(2 ** attempt)
            else:
                raise
    return {}


def parse_response(text, stds_dict):
    """Parse model response into {std_id: expanded_description}."""
    result = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Try to find a standard ID at the start
        for sid in stds_dict:
            if line.startswith(sid):
                desc = line[len(sid):].lstrip(":").strip()
                if desc:
                    result[sid] = desc
                break
    return result


def main():
    all_standards = load_standards(STANDARDS_PATH)
    clusters = get_clusters_with_standards(all_standards)

    print(f"Found {len(clusters)} clusters with 2+ standards")

    expanded = {}
    done = 0

    for cluster_id, stds_dict in sorted(clusters.items()):
        descs = generate_descriptions(cluster_id, stds_dict)
        for sid, desc in descs.items():
            expanded[sid] = desc
        done += 1
        if done % 10 == 0:
            print(f"  Processed {done}/{len(clusters)} clusters ({len(expanded)} descriptions)")

    print(f"\nGenerated {len(expanded)} expanded descriptions")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(expanded, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
