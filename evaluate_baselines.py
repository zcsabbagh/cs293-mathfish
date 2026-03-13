#!/usr/bin/env python3
"""Train and evaluate baseline models on Building On task.

Models: TF-IDF + Logistic Regression, SetFit, DeBERTa-v3-base
"""

import argparse
import json
import csv
import re
import os
import time
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

TRAIN_FILE = "together_train.jsonl"
TEST_FILE = "together_val.jsonl"
GRANULARITIES = ["standard", "cluster", "domain", "grade"]
CCSS_PATTERN = re.compile(r"\b[K\d]+\.\w+\.\w+(?:\.\w+)?\b")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(path: str) -> tuple[list[str], list[list[str]]]:
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            text = entry["text"].removeprefix("<s>").removesuffix("</s>").strip()
            marker = "Building On Standards: "
            idx = text.index(marker)
            problem = text[:idx].removeprefix("Problem: ").strip()
            codes = [c.strip() for c in text[idx + len(marker):].split(",") if c.strip()]
            texts.append(problem)
            labels.append(codes)
    return texts, labels


# ---------------------------------------------------------------------------
# Metrics (same as other eval scripts)
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
        f1s, precisions, recalls, ems = [], [], [], []
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


def print_and_save(model_name: str, metrics: dict):
    print(f"\n  Results for {model_name}:")
    for level in GRANULARITIES:
        m = metrics[level]
        print(f"    {level:10s} F1: {m['f1']:.4f}  P: {m['precision']:.4f}  R: {m['recall']:.4f}  EM: {m['exact_match']:.4f}")

    # Save JSON
    safe = model_name.replace(" ", "_").replace("/", "_").replace("+", "_")
    result = {"model": model_name, "step": "final", "epoch": "final"}
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
    print(f"  Saved to {json_path} and {csv_path}")


# ---------------------------------------------------------------------------
# 1. TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------
def run_tfidf_logreg(train_texts, train_labels, test_texts, test_labels):
    print("\n" + "=" * 60)
    print("Training TF-IDF + Logistic Regression")
    print("=" * 60)

    t0 = time.time()

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_labels)
    print(f"  Labels: {len(mlb.classes_)} unique standards")

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = tfidf.fit_transform(train_texts)
    X_test = tfidf.transform(test_texts)

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, C=1.0, solver="liblinear"),
        n_jobs=1
    )
    clf.fit(X_train, y_train)

    # Label space is very sparse (~6 samples per label), so the model never
    # predicts positive at default threshold. Use top-k instead, where k is
    # the average number of labels per training example.
    avg_k = max(1, round(sum(len(l) for l in train_labels) / len(train_labels)))
    scores = clf.decision_function(X_test)
    predictions = []
    for row in scores:
        top_k = np.argsort(row)[-avg_k:]
        predictions.append(list(mlb.classes_[top_k]))

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    metrics = compute_metrics(predictions, test_labels)
    print_and_save("TF-IDF+LogReg", metrics)
    return metrics


# ---------------------------------------------------------------------------
# 2. SetFit
# ---------------------------------------------------------------------------
def run_setfit(train_texts, train_labels, test_texts, test_labels):
    print("\n" + "=" * 60)
    print("Training SetFit")
    print("=" * 60)

    from setfit import SetFitModel, Trainer, TrainingArguments
    from datasets import Dataset

    t0 = time.time()

    # SetFit expects multi-label as list of ints, so we binarize
    mlb = MultiLabelBinarizer()
    y_train_bin = mlb.fit_transform(train_labels)
    print(f"  Labels: {len(mlb.classes_)} unique standards")

    # Create datasets
    train_ds = Dataset.from_dict({
        "text": train_texts,
        "label": [row.tolist() for row in y_train_bin],
    })
    test_ds = Dataset.from_dict({
        "text": test_texts,
        "label": [row.tolist() for row in mlb.transform(test_labels)],
    })

    model = SetFitModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        multi_target_strategy="one-vs-rest",
        labels=list(mlb.classes_),
    )

    args = TrainingArguments(
        batch_size=64,
        num_epochs=1,
        num_iterations=10,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )
    trainer.train()

    # Predict using top-k (default predict() returns all zeros due to sparse labels)
    avg_k = max(1, round(sum(len(l) for l in train_labels) / len(train_labels)))
    y_proba = model.predict_proba(test_texts)
    if hasattr(y_proba, "numpy"):
        y_proba = y_proba.numpy()
    y_proba = np.array(y_proba)

    predictions = []
    for row in y_proba:
        top_k = np.argsort(row)[-avg_k:]
        predictions.append(list(mlb.classes_[top_k]))

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    metrics = compute_metrics(predictions, test_labels)
    print_and_save("SetFit", metrics)
    return metrics


# ---------------------------------------------------------------------------
# 3. DeBERTa-v3-base
# ---------------------------------------------------------------------------
def run_deberta(train_texts, train_labels, test_texts, test_labels):
    print("\n" + "=" * 60)
    print("Training DeBERTa-v3-base")
    print("=" * 60)

    import torch
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from transformers import AutoModelForSequenceClassification
    from transformers import get_linear_schedule_with_warmup
    from transformers.models.deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")

    t0 = time.time()

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_labels)
    y_test = mlb.transform(test_labels)
    num_labels = len(mlb.classes_)
    print(f"  Labels: {num_labels} unique standards")

    model_name = "microsoft/deberta-v3-base"
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    ).to(device)

    class TextDataset(TorchDataset):
        def __init__(self, texts, labels, tokenizer, max_len=512):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx], truncation=True, padding="max_length",
                max_length=self.max_len, return_tensors="pt"
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.float),
            }

    train_ds = TextDataset(train_texts, y_train, tokenizer)
    test_ds = TextDataset(test_texts, y_test, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    num_epochs = 3
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    # Train
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{num_epochs} — loss: {avg_loss:.4f}")

    # Evaluate using top-k (sigmoid > 0.5 threshold gives all zeros due to sparse labels)
    avg_k = max(1, round(sum(len(l) for l in train_labels) / len(train_labels)))
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.sigmoid().cpu().numpy()
            all_logits.append(logits)

    all_scores = np.vstack(all_logits)
    predictions = []
    for row in all_scores:
        top_k = np.argsort(row)[-avg_k:]
        predictions.append(list(mlb.classes_[top_k]))

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    metrics = compute_metrics(predictions, test_labels)
    print_and_save("DeBERTa-v3-base", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["tfidf", "setfit", "deberta"],
                        choices=["tfidf", "setfit", "deberta"],
                        help="Which models to run")
    parser.add_argument("--train-file", default=TRAIN_FILE)
    parser.add_argument("--test-file", default=TEST_FILE)
    args = parser.parse_args()

    train_texts, train_labels = load_data(args.train_file)
    test_texts, test_labels = load_data(args.test_file)
    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

    if "tfidf" in args.models:
        run_tfidf_logreg(train_texts, train_labels, test_texts, test_labels)

    if "setfit" in args.models:
        run_setfit(train_texts, train_labels, test_texts, test_labels)

    if "deberta" in args.models:
        run_deberta(train_texts, train_labels, test_texts, test_labels)

    print("\nAll done.")


if __name__ == "__main__":
    main()
