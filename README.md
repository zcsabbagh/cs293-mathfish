# MathFish: Hierarchical Math Standard Classification

Predicting Common Core prerequisite ("Building On") standards from math problems using hierarchical tree search and fine-tuned LLMs. Built for CS293.

## Quick Start

```bash
pip install together python-dotenv
echo "TOGETHER_API_KEY=your_key" > .env
python3 scaffold_inference.py 100  # run eval on 100 problems
```

## For Report Writers: Key Documents

Start here for writing the paper:

| File | What's in it |
|---|---|
| **`FINAL.md`** | Complete methodology, all results tables, and key findings. This is the main reference for the report. |
| **`TRIED_FIXES.md`** | Detailed log of all 17+ experiments with hypotheses, results, and verdicts. Use this for the "experiments" or "ablation study" section. |
| **`all_results.json`** | Structured JSON of every experiment's metrics across all models and sample sizes. Use this to generate tables/figures. |

## Results at a Glance

Best fine-tuned (Qwen2.5-14B-Instruct, LoRA SFT, N=250):

| Layer | Accuracy |
|---|---|
| Domain | 83% |
| Cluster | 74% |
| Standard | 60% |

Best non-fine-tuned (Qwen3-80B, N=100, with all optimizations):

| Layer | Accuracy |
|---|---|
| Domain | 81% |
| Cluster | 72% |
| Standard | 51% |

The fine-tuned 14B model outperforms the 80B model by +12% on standard prediction while being 5.7x smaller.

## Repository Structure

### Core Scripts
- `scaffold_inference.py` — Main hierarchical inference and evaluation pipeline
- `create_finetune_data.py` — Generates fine-tuning JSONL from training data
- `dspy_scaffold.py` — DSPy-based prompt optimization (experiment 6, abandoned)

### Data
- `standards.jsonl` — Common Core hierarchy (grades, domains, clusters, standards)
- `coherence_map.jsonl` — Cross-standard prerequisite/follow-up connections
- `together_train.jsonl` / `together_val.jsonl` — Raw train/val problem sets
- `together_mega_train.jsonl` — Fine-tuning training data (3,519 rows)
- `together_mega_eval.jsonl` — Fine-tuning eval data (282 rows)

### Results
- `all_results.json` — All metrics in structured JSON
- `scaffold_inference_*_results.jsonl` — Per-problem predictions for each model run

### Experiments
- `experiments/` — Individual experiment scripts and cached data (example problems, expanded descriptions, etc.)

### Documentation
- `FINAL.md` — Full methodology and results writeup
- `TRIED_FIXES.md` — Experiment log with all 17+ attempted fixes

## Methodology

We use **Tree-NoGrade** mode: the teacher provides the grade level (they know what grade they teach), and the model cascades predictions through Domain → Cluster → Standard. At each layer, predictions are constrained to valid options in the hierarchy.

Key techniques:
- **Student solution generation**: Model solves the problem as a grade-appropriate student before classifying
- **Domain disambiguation**: Targeted revision step for 12 known confusable domain pairs
- **Validation retries**: Retry on invalid outputs with corrective prompts
- **LoRA fine-tuning**: SFT on 1,173 labeled problems ($12 training cost on Together AI)

See `FINAL.md` for the complete methodology and `TRIED_FIXES.md` for the full experiment history.
