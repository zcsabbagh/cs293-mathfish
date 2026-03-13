#!/usr/bin/env python3
"""
Create fine-tuning JSONL for Together AI chat completions format.

For each problem, generates 3 training examples:
  1. Grade + problem → domain
  2. Grade + domain + problem → cluster
  3. Grade + domain + cluster + problem → standard

Uses ground-truth labels at each layer. Optionally generates student solutions
via Together API (slow, ~1239 API calls) or skips them.

Usage:
  python3 create_finetune_data.py                    # train set, no student solutions
  python3 create_finetune_data.py --with-solutions   # train set, generate solutions
  python3 create_finetune_data.py --eval             # val set (100 problems)
"""

import json
import os
import sys
import re
import time
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from together import Together

load_dotenv()
client = Together()

STANDARDS_PATH = "standards.jsonl"
TRAIN_PATH = "together_train.jsonl"
VAL_PATH = "together_val.jsonl"
SOLUTION_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"

# ── Load standards hierarchy (reused from scaffold_inference.py) ──────────────

def load_standards(path):
    standards = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            standards[entry["id"]] = entry
    return standards

def build_hierarchy(standards):
    grades = {}
    domains_by_grade = defaultdict(dict)
    clusters_by_domain = defaultdict(dict)
    standards_by_cluster = defaultdict(dict)

    for sid, entry in standards.items():
        level = entry["level"]
        if level == "Grade":
            grades[sid] = entry["description"]
        elif level == "Domain":
            parent_grade = entry.get("parent", "")
            domains_by_grade[parent_grade][sid] = entry["description"]
        elif level == "Cluster":
            parent_domain = entry.get("parent", "")
            clusters_by_domain[parent_domain][sid] = entry["description"]
        elif level == "Standard":
            parent_cluster = entry.get("parent", "")
            standards_by_cluster[parent_cluster][sid] = entry["description"]

    hs_categories = [sid for sid, e in standards.items() if e["level"] == "HS Category"]
    for cat in hs_categories:
        for domain_id, desc in list(domains_by_grade.get(cat, {}).items()):
            domains_by_grade["HS"][domain_id] = desc
        domains_by_grade.pop(cat, None)

    return grades, domains_by_grade, clusters_by_domain, standards_by_cluster


# ── ID decomposition (reused from scaffold_inference.py) ──────────────────────

def get_grade(std_id):
    if "-" in std_id.split(".")[0]:
        return "HS"
    return std_id.split(".")[0]

def get_domain(std_id):
    parts = std_id.split(".")
    if "-" in parts[0]:
        return parts[0]
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else parts[0]

def get_cluster(std_id):
    parts = std_id.split(".")
    if "-" in parts[0]:
        return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else parts[0]
    if len(parts) >= 3:
        return f"{parts[0]}.{parts[1]}.{parts[2]}"
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else parts[0]

def get_standard(std_id, all_standards):
    """Get the standard-level ID (strip sub-standard suffix)."""
    if std_id in all_standards and all_standards[std_id]["level"] == "Sub-standard":
        return all_standards[std_id].get("parent", std_id)
    return std_id


# ── Parse problems ────────────────────────────────────────────────────────────

SOLUTION_MARKERS = [
    r"Student Response\n.*?(?=\n(?:Anticipated Misconceptions|Activity Synthesis|Student Facing|Are you ready for more\?|Problem \d|\Z))",
    r"Activity Synthesis\n.*?(?=\n(?:Student Facing|Problem \d|\Z))",
    r"Anticipated Misconceptions\n.*?(?=\n(?:Activity Synthesis|Student Response|Student Facing|Problem \d|\Z))",
    r"Extension Student Response\n.*?(?=\n(?:Activity Synthesis|Student Facing|Problem \d|\Z))",
]

def strip_solution(text):
    for pattern in SOLUTION_MARKERS:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def parse_problems(path):
    problems = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            text = entry["text"].replace("<s>", "").replace("</s>", "")
            match = re.search(r"Building On Standards:\s*(.+)$", text, re.MULTILINE)
            if not match:
                continue
            standard_ids = [s.strip() for s in match.group(1).strip().split(",")]
            problem_text = text[: match.start()].strip()
            if problem_text.startswith("Problem: "):
                problem_text = problem_text[len("Problem: "):]
            problems.append({
                "text": strip_solution(problem_text),
                "standard_ids": standard_ids,
            })
    return problems


# ── Prompt builders (same as scaffold_inference.py) ───────────────────────────

def fmt(options):
    return "\n".join(f"  - {k}: {v}" for k, v in sorted(options.items()))

def prompt_domain(problem, grade, domains, student_solution=""):
    solution_section = ""
    if student_solution:
        solution_section = f"\n\n## How a Grade {grade} Student Would Solve This\n{student_solution}"
    return f"""You are a math education expert. Given a math problem and its grade level, predict which domain it belongs to.

## Grade Level
{grade}

## Available Domains at This Grade Level
{fmt(domains)}

## Problem
{problem}{solution_section}

## Instructions
Respond with ONLY the domain ID (e.g., "3.NF", "K.OA", "A-CED"). No explanation."""

def prompt_cluster(problem, grade, domain, domain_desc, clusters, student_solution=""):
    solution_section = ""
    if student_solution:
        solution_section = f"\n\n## How a Grade {grade} Student Would Solve This\n{student_solution}"
    return f"""You are a math education expert. Given a math problem, its grade level, and domain, predict which cluster it belongs to.

## Grade Level
{grade}

## Domain
{domain}: {domain_desc}

## Available Clusters in This Domain
{fmt(clusters)}

## Problem
{problem}{solution_section}

## Instructions
Respond with ONLY the cluster ID (e.g., "3.NF.A", "K.OA.A", "A-CED.A"). No explanation."""

def prompt_standard(problem, grade, domain, domain_desc, cluster, cluster_desc, stds, student_solution=""):
    return f"""You are a math education expert. Given a math problem, its grade level, domain, and cluster, predict the specific standard it aligns to.

## Grade Level
{grade}

## Domain
{domain}: {domain_desc}

## Cluster
{cluster}: {cluster_desc}

## Available Standards in This Cluster
{fmt(stds)}

## Problem
{problem}{f'''

## How a Grade {grade} Student Would Solve This
{student_solution}''' if student_solution else ''}

## Instructions
This problem may align to one or more standards. Respond with the standard ID(s) that best match, comma-separated if multiple (e.g., "3.NF.A.1" or "8.F.B.4, 8.F.B.5"). No explanation."""


# ── Student solution generation ───────────────────────────────────────────────

def generate_student_solution(problem, grade, max_retries=5):
    prompt = f"""You are a grade {grade} math student. Solve the following problem step by step, showing your work the way a {grade} grader would. Use the math vocabulary and techniques appropriate for grade {grade}. Keep it brief (3-5 steps max).

## Problem
{problem}

## Your Solution"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=SOLUTION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(2 ** attempt)
            else:
                raise
    return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Generate eval set instead of train")
    parser.add_argument("--with-solutions", action="store_true", help="Generate student solutions (slow)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of problems")
    args = parser.parse_args()

    all_standards = load_standards(STANDARDS_PATH)
    grades, domains_by_grade, clusters_by_domain, standards_by_cluster = build_hierarchy(all_standards)

    input_path = VAL_PATH if args.eval else TRAIN_PATH
    problems = parse_problems(input_path)
    if args.limit:
        problems = problems[:args.limit]

    suffix = "eval" if args.eval else "train"
    out_path = f"together_mega_{suffix}.jsonl"

    print(f"Parsed {len(problems)} problems from {input_path}")

    # Optionally generate student solutions in parallel
    solutions = {}
    if args.with_solutions:
        print(f"Generating student solutions for {len(problems)} problems...")
        def gen_sol(idx_prob):
            idx, prob = idx_prob
            grade = get_grade(prob["standard_ids"][0])
            sol = generate_student_solution(prob["text"], grade)
            return idx, sol

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(gen_sol, (i, p)): i for i, p in enumerate(problems)}
            for fut in as_completed(futures):
                idx, sol = fut.result()
                solutions[idx] = sol
                if (idx + 1) % 50 == 0:
                    print(f"  Generated {idx + 1}/{len(problems)} solutions")
        print(f"  Done: {len(solutions)} solutions generated")

    # Generate fine-tuning rows
    rows = []
    skipped = 0
    for i, prob in enumerate(problems):
        text = prob["text"]
        true_ids = prob["standard_ids"]
        # Use first standard ID for ground-truth path
        true_id = true_ids[0]

        grade = get_grade(true_id)
        domain = get_domain(true_id)
        cluster = get_cluster(true_id)
        standard = get_standard(true_id, all_standards)

        student_sol = solutions.get(i, "")

        # Look up hierarchy
        avail_domains = domains_by_grade.get(grade, {})
        domain_desc = avail_domains.get(domain, "")
        avail_clusters = clusters_by_domain.get(domain, {})
        cluster_desc = avail_clusters.get(cluster, "")
        avail_stds = standards_by_cluster.get(cluster, {})

        # Skip if any layer is missing from hierarchy
        if not avail_domains or domain not in avail_domains:
            skipped += 1
            continue
        if not avail_clusters or cluster not in avail_clusters:
            skipped += 1
            continue
        if not avail_stds or standard not in avail_stds:
            skipped += 1
            continue

        # 1. Domain classification
        user_msg_domain = prompt_domain(text, grade, avail_domains, student_solution=student_sol)
        rows.append({"messages": [
            {"role": "user", "content": user_msg_domain},
            {"role": "assistant", "content": domain},
        ]})

        # 2. Cluster classification
        user_msg_cluster = prompt_cluster(text, grade, domain, domain_desc, avail_clusters, student_solution=student_sol)
        rows.append({"messages": [
            {"role": "user", "content": user_msg_cluster},
            {"role": "assistant", "content": cluster},
        ]})

        # 3. Standard classification
        # For multi-label, use comma-separated true standards that are in avail_stds
        true_stds_in_cluster = [get_standard(tid, all_standards) for tid in true_ids
                                if get_standard(tid, all_standards) in avail_stds]
        # Deduplicate while preserving order
        seen = set()
        true_stds_unique = []
        for s in true_stds_in_cluster:
            if s not in seen:
                seen.add(s)
                true_stds_unique.append(s)
        answer_std = ", ".join(true_stds_unique) if true_stds_unique else standard

        user_msg_std = prompt_standard(text, grade, domain, domain_desc, cluster, cluster_desc, avail_stds, student_solution=student_sol)
        rows.append({"messages": [
            {"role": "user", "content": user_msg_std},
            {"role": "assistant", "content": answer_std},
        ]})

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"  ({len(rows) // 3} problems × 3 layers)")
    if skipped:
        print(f"  Skipped {skipped} problems (missing from hierarchy)")


if __name__ == "__main__":
    main()
