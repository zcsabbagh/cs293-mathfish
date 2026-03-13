#!/usr/bin/env python3
"""
DSPy-optimized scaffold inference for math standard classification.

Uses BootstrapFewShot to optimize few-shot examples for each layer
of the hierarchy: Grade → Domain → Cluster → Standard.

Usage:
  python3 dspy_scaffold.py optimize [N_train] [N_val]  # optimize prompts
  python3 dspy_scaffold.py eval [N]                     # eval with saved prompts
"""

import json
import os
import sys
import re
import time
from collections import defaultdict
from dotenv import load_dotenv

import dspy

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
TOGETHER_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
STANDARDS_PATH = "standards.jsonl"
TRAIN_PATH = "together_train.jsonl"
VAL_PATH = "together_val.jsonl"
COHERENCE_PATH = "coherence_map.jsonl"
OPTIMIZED_PATH = "dspy_optimized.json"

# ── Reuse hierarchy / parsing from scaffold_inference ─────────────────────────

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
            domains_by_grade[entry.get("parent", "")][sid] = entry["description"]
        elif level == "Cluster":
            clusters_by_domain[entry.get("parent", "")][sid] = entry["description"]
        elif level == "Standard":
            standards_by_cluster[entry.get("parent", "")][sid] = entry["description"]

    hs_categories = [sid for sid, e in standards.items() if e["level"] == "HS Category"]
    for cat in hs_categories:
        for domain_id, desc in list(domains_by_grade.get(cat, {}).items()):
            domains_by_grade["HS"][domain_id] = desc
        domains_by_grade.pop(cat, None)

    return grades, domains_by_grade, clusters_by_domain, standards_by_cluster


SOLUTION_MARKERS = [
    r"Student Response\n.*?(?=\n(?:Anticipated Misconceptions|Activity Synthesis|Student Facing|Are you ready for more\?|Problem \d|\Z))",
    r"Activity Synthesis\n.*?(?=\n(?:Student Facing|Problem \d|\Z))",
    r"Anticipated Misconceptions\n.*?(?=\n(?:Activity Synthesis|Student Response|Student Facing|Problem \d|\Z))",
    r"Extension Student Response\n.*?(?=\n(?:Activity Synthesis|Student Facing|Problem \d|\Z))",
]


def strip_solution(text):
    for pattern in SOLUTION_MARKERS:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


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


def fmt(options):
    return "\n".join(f"  - {k}: {v}" for k, v in sorted(options.items()))


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
            problem_text = text[:match.start()].strip()
            if problem_text.startswith("Problem: "):
                problem_text = problem_text[len("Problem: "):]
            problems.append({
                "text": problem_text,
                "standard_ids": standard_ids,
            })
    return problems


# ── Load data ─────────────────────────────────────────────────────────────────

all_standards = load_standards(STANDARDS_PATH)
GRADES, DOMAINS_BY_GRADE, CLUSTERS_BY_DOMAIN, STANDARDS_BY_CLUSTER = build_hierarchy(all_standards)


# ── Load coherence map ────────────────────────────────────────────────────────

def _normalize_edge_id(eid):
    """Convert coherence map IDs (e.g., '3.NF.1') to our standard IDs (e.g., '3.NF.A.1')."""
    eid = eid.split("||")[0].strip().split(",")[0].strip()
    if eid.startswith("0."):
        eid = "K." + eid[2:]
    parts = eid.split(".")
    if len(parts) == 4 and len(parts[3]) == 1 and parts[3].isalpha():
        eid = f"{parts[0]}.{parts[1]}.{parts[2]}{parts[3]}"
    return eid


def _to_short(full_id):
    """Strip cluster letter: 3.NF.A.1 -> 3.NF.1, A-CED.A.1 -> A-CED.1"""
    parts = full_id.split(".")
    if "-" in parts[0]:
        return f"{parts[0]}.{parts[2]}" if len(parts) >= 3 else full_id
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.{parts[3]}"
    if len(parts) == 3:
        return f"{parts[0]}.{parts[1]}"
    return full_id


def load_coherence(path):
    """Load coherence edges and map to our standard IDs. Returns {std_id: set of connected std_ids}."""
    data = json.load(open(path))
    edges = data["coherence_edges"]

    # Build short->full map
    short_to_full = defaultdict(list)
    for sid, entry in all_standards.items():
        if entry["level"] in ("Standard", "Sub-standard"):
            short_to_full[_to_short(sid)].append(sid)

    coherence = defaultdict(set)
    for e in edges:
        from_full = short_to_full.get(_normalize_edge_id(e["from"]), [])
        to_full = short_to_full.get(_normalize_edge_id(e["to"]), [])
        if from_full and to_full:
            for f in from_full:
                for t in to_full:
                    coherence[f].add(t)
                    coherence[t].add(f)
    return dict(coherence)


COHERENCE = load_coherence(COHERENCE_PATH)


# ── DSPy Signatures ──────────────────────────────────────────────────────────

class PredictDomain(dspy.Signature):
    """You are a math education expert. Given a math problem and its grade level, predict which Common Core math domain it belongs to. Respond with ONLY the domain ID."""
    problem = dspy.InputField(desc="The math problem text")
    grade = dspy.InputField(desc="The grade level (e.g., '3', 'K', 'HS')")
    available_domains = dspy.InputField(desc="List of available domains with descriptions")
    domain = dspy.OutputField(desc="The domain ID only (e.g., '3.NF', 'K.OA', 'A-CED')")


class PredictCluster(dspy.Signature):
    """You are a math education expert. Given a math problem, its grade level, and domain, predict which Common Core cluster it belongs to. Respond with ONLY the cluster ID."""
    problem = dspy.InputField(desc="The math problem text")
    grade = dspy.InputField(desc="The grade level")
    domain_info = dspy.InputField(desc="The domain ID and description")
    available_clusters = dspy.InputField(desc="List of available clusters with descriptions")
    cluster = dspy.OutputField(desc="The cluster ID only (e.g., '3.NF.A', 'K.OA.A')")


class PredictStandard(dspy.Signature):
    """You are a math education expert. Given a math problem, its grade, domain, cluster, and coherence connections between standards, predict the specific Common Core standard. Respond with ONLY the standard ID."""
    problem = dspy.InputField(desc="The math problem text")
    grade = dspy.InputField(desc="The grade level")
    domain_info = dspy.InputField(desc="The domain ID and description")
    cluster_info = dspy.InputField(desc="The cluster ID and description")
    available_standards = dspy.InputField(desc="List of available standards with descriptions")
    coherence_hints = dspy.InputField(desc="Standards connected via the Common Core coherence map (prerequisite/follow-up relationships)")
    standard = dspy.OutputField(desc="The standard ID only (e.g., '3.NF.A.1', 'A-CED.A.1')")


# ── DSPy Module: Full cascade ────────────────────────────────────────────────

class StandardClassifier(dspy.Module):
    def __init__(self):
        self.predict_domain = dspy.Predict(PredictDomain)
        self.predict_cluster = dspy.Predict(PredictCluster)
        self.predict_standard = dspy.Predict(PredictStandard)

    def forward(self, problem, grade):
        # Domain
        avail_domains = DOMAINS_BY_GRADE.get(grade, {})
        if not avail_domains:
            return dspy.Prediction(domain="UNKNOWN", cluster="UNKNOWN", standard="UNKNOWN")

        if len(avail_domains) == 1:
            pred_domain = next(iter(avail_domains))
        else:
            domain_result = self.predict_domain(
                problem=problem,
                grade=grade,
                available_domains=fmt(avail_domains),
            )
            pred_domain = domain_result.domain.strip().strip('"').strip("'")

        # Cluster
        domain_desc = avail_domains.get(pred_domain, "")
        avail_clusters = CLUSTERS_BY_DOMAIN.get(pred_domain, {})
        if not avail_clusters:
            return dspy.Prediction(domain=pred_domain, cluster="UNKNOWN", standard="UNKNOWN")

        if len(avail_clusters) == 1:
            pred_cluster = next(iter(avail_clusters))
        else:
            cluster_result = self.predict_cluster(
                problem=problem,
                grade=grade,
                domain_info=f"{pred_domain}: {domain_desc}",
                available_clusters=fmt(avail_clusters),
            )
            pred_cluster = cluster_result.cluster.strip().strip('"').strip("'")

        # Standard
        cluster_desc = avail_clusters.get(pred_cluster, "")
        avail_stds = STANDARDS_BY_CLUSTER.get(pred_cluster, {})
        if not avail_stds:
            return dspy.Prediction(domain=pred_domain, cluster=pred_cluster, standard="UNKNOWN")

        if len(avail_stds) == 1:
            pred_standard = next(iter(avail_stds))
        else:
            # Build coherence hints for available standards
            hints = []
            for std_id in sorted(avail_stds.keys()):
                connected = COHERENCE.get(std_id, set())
                if connected:
                    conns = ", ".join(sorted(connected)[:5])
                    hints.append(f"  {std_id} connects to: {conns}")
            coherence_str = "\n".join(hints) if hints else "No coherence data available"

            std_result = self.predict_standard(
                problem=problem,
                grade=grade,
                domain_info=f"{pred_domain}: {domain_desc}",
                cluster_info=f"{pred_cluster}: {cluster_desc}",
                available_standards=fmt(avail_stds),
                coherence_hints=coherence_str,
            )
            pred_standard = std_result.standard.strip().strip('"').strip("'")

        return dspy.Prediction(domain=pred_domain, cluster=pred_cluster, standard=pred_standard)


# ── Build training examples ──────────────────────────────────────────────────

def build_dspy_examples(problems, n=None):
    """Convert parsed problems into DSPy Examples for the full cascade."""
    examples = []
    for p in (problems[:n] if n else problems):
        std_id = p["standard_ids"][0]
        grade = get_grade(std_id)
        domain = get_domain(std_id)
        cluster = get_cluster(std_id)

        # Only include if hierarchy is valid
        if (grade in GRADES or grade == "HS") and \
           domain in DOMAINS_BY_GRADE.get(grade, {}) and \
           cluster in CLUSTERS_BY_DOMAIN.get(domain, {}):
            ex = dspy.Example(
                problem=p["text"],
                grade=grade,
                domain=domain,
                cluster=cluster,
                standard=std_id,
            ).with_inputs("problem", "grade")
            examples.append(ex)
    return examples


# ── Metric ────────────────────────────────────────────────────────────────────

def standard_accuracy(example, prediction, trace=None):
    """Check if predicted standard matches the true standard."""
    return prediction.standard.strip().strip('"').strip("'") == example.standard


def cascade_accuracy(example, prediction, trace=None):
    """Weighted accuracy: domain=0.2, cluster=0.3, standard=0.5"""
    pred_d = prediction.domain.strip().strip('"').strip("'")
    pred_c = prediction.cluster.strip().strip('"').strip("'")
    pred_s = prediction.standard.strip().strip('"').strip("'")

    score = 0.0
    if pred_d == example.domain:
        score += 0.2
    if pred_c == example.cluster:
        score += 0.3
    if pred_s == example.standard:
        score += 0.5
    return score


# ── Main ──────────────────────────────────────────────────────────────────────

def optimize(n_train=200, n_val=50):
    """Optimize the classifier using BootstrapFewShot."""
    print(f"Loading data...")
    train_problems = parse_problems(TRAIN_PATH)
    val_problems = parse_problems(VAL_PATH)

    train_examples = build_dspy_examples(train_problems, n_train)
    val_examples = build_dspy_examples(val_problems, n_val)
    print(f"Train examples: {len(train_examples)}, Val examples: {len(val_examples)}")

    # Configure DSPy with Together AI
    lm = dspy.LM(
        f"together_ai/{TOGETHER_MODEL}",
        api_key=os.environ["TOGETHER_API_KEY"],
        max_tokens=1024,
        temperature=0.0,
    )
    dspy.configure(lm=lm)

    classifier = StandardClassifier()

    # Optimize with BootstrapFewShot
    optimizer = dspy.BootstrapFewShot(
        metric=standard_accuracy,
        max_bootstrapped_demos=5,
        max_labeled_demos=5,
        max_rounds=2,
    )

    print(f"\nOptimizing with BootstrapFewShot...")
    print(f"  max_bootstrapped_demos=5, max_labeled_demos=5, max_rounds=2")
    optimized = optimizer.compile(classifier, trainset=train_examples)

    # Save
    optimized.save(OPTIMIZED_PATH)
    print(f"\nSaved optimized program to {OPTIMIZED_PATH}")

    # Evaluate
    print(f"\nEvaluating on {len(val_examples)} val examples...")
    evaluate(val_examples, optimized)


def evaluate(examples=None, classifier=None, n=100):
    """Evaluate the classifier on val data."""
    if examples is None:
        val_problems = parse_problems(VAL_PATH)
        examples = build_dspy_examples(val_problems, n)

    if classifier is None:
        lm = dspy.LM(
            f"together_ai/{TOGETHER_MODEL}",
            api_key=os.environ["TOGETHER_API_KEY"],
            max_tokens=1024,
            temperature=0.0,
        )
        dspy.configure(lm=lm)
        classifier = StandardClassifier()
        classifier.load(OPTIMIZED_PATH)
        print(f"Loaded optimized program from {OPTIMIZED_PATH}")

    correct = {"domain": 0, "cluster": 0, "standard": 0}
    total = len(examples)

    for i, ex in enumerate(examples):
        try:
            pred = classifier(problem=ex.problem, grade=ex.grade)
            pred_d = pred.domain.strip().strip('"').strip("'")
            pred_c = pred.cluster.strip().strip('"').strip("'")
            pred_s = pred.standard.strip().strip('"').strip("'")

            d_ok = pred_d == ex.domain
            c_ok = pred_c == ex.cluster
            s_ok = pred_s == ex.standard

            if d_ok: correct["domain"] += 1
            if c_ok: correct["cluster"] += 1
            if s_ok: correct["standard"] += 1

            mark = "✓" if s_ok else "✗"
            print(f"[{i+1}/{total}] True: {ex.standard:15s} | "
                  f"Pred: D={pred_d:8s} C={pred_c:10s} S={pred_s:15s} | {mark}")
        except Exception as e:
            print(f"[{i+1}/{total}] ERROR: {e}")

    print(f"\nResults over {total} problems:")
    print(f"  Domain:    {correct['domain']:4d}/{total}  ({100*correct['domain']/total:.1f}%)")
    print(f"  Cluster:   {correct['cluster']:4d}/{total}  ({100*correct['cluster']/total:.1f}%)")
    print(f"  Standard:  {correct['standard']:4d}/{total}  ({100*correct['standard']/total:.1f}%)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "optimize"

    if cmd == "optimize":
        n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        optimize(n_train, n_val)
    elif cmd == "eval":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        evaluate(n=n)
    else:
        print(f"Usage: python3 {sys.argv[0]} [optimize|eval] [args...]")
