# Scaffold Inference: Hierarchical Math Standard Classification

## Task

Given a math problem from the MathFish dataset, predict which Common Core State Standard (CCSS) it aligns to. Specifically, we predict the **"Building On"** (prerequisite) standard — the prior knowledge a student needs to solve the problem. This serves a teacher use case: "This 8th grader doesn't understand topic X — what did they miss in an earlier grade?"

The CCSS hierarchy has 4 levels:
- **Grade** (K, 1–8, HS)
- **Domain** (e.g., 6.EE = Expressions & Equations)
- **Cluster** (e.g., 6.EE.B = Reason about and solve one-variable equations)
- **Standard** (e.g., 6.EE.B.7 = Solve real-world problems by writing equations)

Some standards also have **sub-standards** (e.g., 6.RP.A.3c), and 41% of problems have multiple valid labels.

## Methodology

### Tree-NoGrade Cascading Inference

We use a **hierarchical tree search** that predicts each layer conditioned on the previous:

```
Grade (given) → Domain → Cluster → Standard
```

**Why skip grade prediction?** Teachers know what grade they teach. Providing ground-truth grade eliminates a source of cascading error and reflects the real use case — a teacher asking "what prerequisite is this student missing?" already knows the grade context.

At each layer, the model receives:
1. The math problem text
2. The predicted (or ground-truth) values from previous layers
3. The **valid options** at this point in the hierarchy (e.g., only domains within the given grade)

This constrains the search space at each step and prevents impossible predictions.

### Key Optimizations

- **Single-option pruning**: When only one option exists at a layer (e.g., a cluster with one standard), skip the API call and auto-select. Saves cost and prevents hallucination errors.
- **Validation retries**: If the model outputs an invalid ID, retry up to 3 times with a corrective prompt listing valid options.
- **Multi-label evaluation**: 41% of problems have multiple valid standards. A prediction is correct if it matches *any* valid label. We report three metrics:
  - **Strict**: First prediction matches any true label
  - **Multi-pred**: Any of the model's comma-separated predictions matches any true label
  - **Parent-match**: Additionally counts parent/sub-standard matches (e.g., `6.RP.A.3` matches `6.RP.A.3c`)
- **Student solution generation**: Before classifying, prompt the model to solve the problem as a grade-appropriate student. The generated solution reveals mathematical vocabulary and operations that help identify the correct standard.
- **Domain disambiguation**: After domain prediction, if the result falls into a known confusable pair (12 pairs identified from error analysis, e.g., 7.EE vs 7.G), a follow-up call asks the model to reconsider with a brief distinction tip.

### Prompt Engineering Ceiling

We ran 17 experiments total (see TRIED_FIXES.md). Experiments 10–17 all attempted to improve classification by modifying prompt content: adding reasoning examples, expanding descriptions, reordering candidates, adding revision steps, using coherence maps, generating example problems. **All either had no effect or hurt performance.** The model reached its prompt-engineering ceiling, motivating fine-tuning.

## Fine-Tuning

### Data

- **Training**: 3,519 rows from 1,173 problems (3 rows per problem: domain, cluster, standard classification)
- **Eval**: 282 rows from 94 problems
- **Format**: Together AI chat completions (`{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`)
- Ground-truth labels used at each layer (teacher forcing)
- No student solutions in training data

### Models

- **Qwen3-8B-Base** (LoRA SFT, 12 epochs): Failed — base model couldn't learn chat format from 3.5K examples. Output gibberish with chat API; produced valid but poor results with completions API (30% standard).
- **Qwen2.5-14B-Instruct** (LoRA SFT): Succeeded — instruct model already knows instruction following, so fine-tuning focused on the classification task.

LoRA config: rank 64, alpha 128, LR 1e-5, cosine scheduler, batch size 16, 12 epochs. Cost: ~$12.

## Results

### Best Non-Fine-Tuned: Qwen3-80B-A3B-Instruct

With student solutions + domain disambiguation (N=100):

| Layer | Strict | Multi-pred | Parent-match |
|-------|--------|------------|--------------|
| Grade | 100% | 100% | 100% |
| Domain | 81% | 81% | 81% |
| Cluster | 72% | 72% | 72% |
| Standard | **51%** | **61%** | **69%** |

250-sample baseline (with solutions + disambiguation):

| Layer | Strict | Multi-pred | Parent-match |
|-------|--------|------------|--------------|
| Domain | 79% | — | — |
| Cluster | 68% | — | — |
| Standard | **48%** | **56%** | **62%** |

### Best Fine-Tuned: Qwen2.5-14B-Instruct (LoRA SFT)

Without student solutions, without domain disambiguation (N=250):

| Layer | Strict | Multi-pred | Parent-match |
|-------|--------|------------|--------------|
| Grade | 100% | 100% | 100% |
| Domain | **83%** | — | — |
| Cluster | **74%** | — | — |
| Standard | **60%** | **60%** | **67%** |

Full results across sample sizes (no solutions, no disambiguation):

| N | Domain | Cluster | Std Strict | Std Multi | Std Parent |
|---|--------|---------|------------|-----------|------------|
| 100 | 81% | 71% | 56% | 57% | 66% |
| 250 | 83% | 74% | 60% | 60% | 67% |
| 500 | 82% | 72% | 54% | 54% | 63% |

With student solutions + domain disambiguation (N=500):

| N | Domain | Cluster | Std Strict | Std Multi | Std Parent |
|---|--------|---------|------------|-----------|------------|
| 100 | 80% | 70% | 54% | 54% | 63% |
| 250 | 83% | 73% | 58% | 57% | 64% |
| 500 | 82% | 73% | 53% | 53% | 62% |

Note: Adding solutions at inference time slightly hurt the fine-tuned model because training data did not include solutions (distribution shift).

### Comparison Summary (N=250)

| Configuration | Domain | Cluster | Std (strict) |
|---|---|---|---|
| 80B instruct + solutions + disambig | 79% | 68% | 48% |
| **14B fine-tuned (no solutions)** | **83%** | **74%** | **60%** |
| 14B fine-tuned + solutions + disambig | 83% | 73% | 58% |
| 8B base fine-tuned | 72% | 49% | 30% |
| 7B instruct (no fine-tuning) | 70% | 57% | 39% |

**The fine-tuned 14B instruct model outperforms the 80B instruct model by +4% domain, +6% cluster, and +12% standard accuracy** — while being 5.7x smaller. Fine-tuning on just 1,173 task-specific examples with LoRA ($12 training cost) was more effective than all 17 prompt engineering experiments combined.

### Small Model Baselines (N=100, no fine-tuning, no solutions)

| Model | Domain | Cluster | Std (strict) |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 70% | 57% | 39% |
| Gemma-3n-E4B | 45% | 26% | 12% |
| Qwen2-1.5B-Instruct | 29% | 13% | 4% |

## Key Findings

1. **Hierarchical tree search works**: Constraining predictions at each layer to valid options in the hierarchy prevents impossible outputs and improves accuracy over flat classification.

2. **Grade prediction is unnecessary for the teacher use case**: Teachers know their grade level. Providing ground-truth grade eliminates cascading errors from incorrect grade prediction.

3. **Prompt engineering has a ceiling**: After the initial optimizations (student solutions +3%, domain disambiguation +3%), seven additional prompt modifications all failed. The model's ability to classify standards is bottlenecked by domain understanding, not prompt format.

4. **Fine-tuning a smaller model beats prompting a larger one**: A 14B model fine-tuned on 1,173 examples outperforms an 80B model with optimized prompts. Task-specific training is more effective than general capability for constrained classification.

5. **Training/inference distribution match matters**: Student solutions helped the 80B model but slightly hurt the fine-tuned 14B because training data lacked solutions. Training with solutions would likely improve results further.

6. **Base models need instruct foundations**: Fine-tuning Qwen3-8B-Base failed because 3.5K examples weren't enough to teach both instruction following and the classification task. Starting from an instruct model lets fine-tuning focus on the task.

## Files

- `scaffold_inference.py` — Main inference and evaluation script
- `create_finetune_data.py` — Generates fine-tuning JSONL from training data
- `together_mega_train.jsonl` — Fine-tuning training data (3,519 rows)
- `together_mega_eval.jsonl` — Fine-tuning eval data (282 rows)
- `all_results.json` — All experiment results in structured JSON
- `TRIED_FIXES.md` — Detailed log of all 17+ experiments
- `standards.jsonl` — CCSS hierarchy (grades, domains, clusters, standards)
- `coherence_map.jsonl` — Cross-standard prerequisite/follow-up connections
- `experiments/` — Scripts and cached data for individual experiments
