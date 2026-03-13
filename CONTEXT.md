Here's the context to paste in:

---

This is a Stanford course project for CS 293 (AI for Education). We're on Assignment 3, building toward a final paper. The project uses the MathFish dataset (allenai/mathfish), which contains ~13,065 K-12 math problems labeled with Common Core State Standards (CCSS) codes. Each problem can have multiple label types: "Addressing" (what standard the problem directly tests) and "Building On" (prerequisite standards a student needs to already know before attempting the problem). The MathFish paper by Lucy et al. only modeled the Addressing relation. Our contribution is modeling the Building On relation, which is a harder task because prerequisite standards are never explicitly mentioned in the problem text — the model has to learn an implicit pedagogical dependency structure.

In our A3 submission, we trained Llama models (1B, 3B, 8B) and RoBERTa on both tasks. Key finding was that model size matters more than training duration for this task (8B at 3 epochs beat 1B at 8 epochs by 2.7x on Addressing). But the results were scattered across different configurations and the grader said the performance enhancements were under-motivated. For the final paper, we're narrowing focus to Building On only and running one clean experiment.

The experiment: fine-tune Qwen3 base models (0.6B, 1.7B, and we'd like larger sizes too if available — 4B, 8B) on the Building On task using Together AI's fine-tuning API. We train each model for 12 epochs, saving a checkpoint every epoch (n_checkpoints=12). Then we evaluate every checkpoint on the full Building On test set (~1,739 problems), reporting standard-level F1, cluster F1, domain F1, and grade F1. This gives us a clean 2D figure: x-axis is epoch, y-axis is standard-level F1, one curve per model size. The hypothesis is that larger models converge faster and to a higher ceiling, and that the gap between model sizes is larger than the gap between early and late checkpoints of the same model.

The dataset is structured like this, one entry per problem:

```json
{
  "id": "fl_problem_001090",
  "text": "Find a single integer that represents...",
  "standards": [["Building On", "5.NF.A.2"], ["Building On", "1.OA.B.3"]],
  "source": "Fishtank Learning"
}
```

For fine-tuning, we're using base (not instruction-tuned) models and formatting the training data as plain text completion:

```jsonl
{"text": "<s>Problem: {problem text}\n\nBuilding On Standards: {comma-separated CCSS codes}</s>"}
```

No standard descriptions in the prompt — we already tested description conditioning in A3 and it barely helped Building On (moved from 0.040 to 0.066). Keeping the format simple so the only experimental variables are model size and training time.

The dataset split is approximately 2,843 train / 1,739 test for Building On, after filtering out problems with empty Building On annotations. We previously evaluated on a 200-problem subset for cost reasons (frontier API calls were expensive), but since we're now running local inference on fine-tuned models, we evaluate on the full test set. CCSS has four granularity levels: Standard (e.g., 3.OA.A.1), Cluster (e.g., 3.OA.A), Domain (e.g., 3.OA), and Grade (e.g., 3). We report F1 at all four levels. Each problem maps to 1-5 CCSS codes.
