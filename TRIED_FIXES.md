# Tried Fixes — Scaffold Inference

## Baseline
- Model: Qwen/Qwen3-Next-80B-A3B-Instruct
- Task: 4-layer hierarchical classification (Grade → Domain → Cluster → Standard)
- Eval: 100 validation problems, Tree-NoGrade mode (ground-truth grade, cascade rest)

**Baseline results (original prompts, no optimizations):**
| Layer    | Accuracy |
|----------|----------|
| Grade    | 100%     |
| Domain   | ~75%     |
| Cluster  | ~68%     |
| Standard | ~42%     |

---

## 1. Constrained prompts ("MUST select from list")
**Hypothesis:** Adding explicit constraint language would reduce hallucinated options.
**Change:** Added "You MUST select from the list below" to all prompts.
**Result:** Standard accuracy dropped (Llama: 47% → 36% oracle). Slightly hurt performance.
**Verdict:** ❌ Reverted. The model already follows the list; adding pressure confused it.

## 2. With vs. without solution content
**Hypothesis:** Including the sample solution would give the model more signal.
**Change:** Compared prompts with full text (problem + solution) vs. problem-only.
**Result:** 0-2% difference across all layers. Negligible impact.
**Verdict:** ➖ No effect. Kept solutions in by default.

## 3. Multi-label evaluation fix
**Hypothesis:** 41% of val problems have multiple true standards spanning different grades/domains. Only checking `true_ids[0]` penalized correct predictions.
**Change:** Computed `true_grades = all_true_values(true_ids, get_grade)` and checked `pred in true_grades` at each layer.
**Result:** Qwen grade accuracy improved 25% → 34% (in full tree mode). Domain/cluster also improved slightly.
**Verdict:** ✅ Kept. Was a bug fix, not a model change.

## 4. Single-option pruning
**Hypothesis:** When only one option exists at a layer (e.g., a cluster with 1 standard), skip the API call and auto-select.
**Change:** Added `if len(avail_X) == 1: pred = next(iter(avail_X))` at each layer.
**Result:** Eliminated a class of errors where the model hallucinated outside a 1-option set. Small accuracy boost.
**Verdict:** ✅ Kept. Saves API calls and prevents impossible errors.

## 5. Sub-standard retry (3 attempts)
**Hypothesis:** Many "errors" are the model predicting parent standard (e.g., `6.SP.B.5`) when true label is sub-standard (e.g., `6.SP.B.5d`). A follow-up prompt asking to pick the specific sub-standard would fix these.
**Change:** After layer 4, if prediction has sub-standards, prompt model to pick one (up to 3 retries).
**Result:** Standard accuracy **dropped** from 44% → 35%.
**Root cause:** When the true label IS the parent standard, the retry forces a sub-standard pick, turning correct answers into wrong ones.
**Verdict:** ❌ Reverted.

## 5b. Sub-standard retry with parent as option
**Hypothesis:** Fix 5 by including the parent standard as a choice alongside its sub-standards.
**Change:** Added parent to the option list in the sub-standard prompt so model can choose to stay at parent level.
**Result:** 35% → 40%. Better than 5, but still below the 44% baseline without retry.
**Root cause:** Model still tends to drill into sub-standards when it shouldn't. Net negative vs. no retry.
**Verdict:** ❌ Reverted.

---

## Current best results (kept fixes: multi-label eval + single-option pruning)
| Layer    | Accuracy |
|----------|----------|
| Grade    | 100%     |
| Domain   | 77%      |
| Cluster  | 70%      |
| Standard | 44%      |

## Key bottlenecks
1. **Domain errors (~23%)**: The model frequently confuses related domains within a grade (e.g., 6.EE vs 6.G, 3.MD vs 3.NF). These cascade into cluster and standard errors.
2. **"Building On" standards**: Problems reference standards from earlier grades, which confuses grade prediction in full tree mode.
3. **Sub-standard granularity**: The distinction between sub-standards (e.g., `6.SP.B.5c` vs `6.SP.B.5d`) is very fine-grained and hard to get right.

## 6. DSPy prompt optimization (BootstrapFewShot)
**Hypothesis:** DSPy's automatic prompt optimization with few-shot example selection would outperform hand-crafted prompts.
**Change:** Built `dspy_scaffold.py` with DSPy signatures for each layer. Tried:
- `ChainOfThought` predictors with BootstrapFewShot (3 demos, 1 round): Standard 37.4%
- `Predict` predictors with BootstrapFewShot (5 demos, 2 rounds): Standard 38.8%
- `Predict` + coherence map as additional context (5 demos, 2 rounds): Standard 39.4%
**Result:** All DSPy variants underperformed the hand-crafted baseline (44%). Best DSPy: 39.4%.
**Root cause:** DSPy is not well-suited for this task because:
1. The option set changes every problem (different domains/clusters/standards depending on hierarchy position), so few-shot demos from one part of the hierarchy don't transfer
2. Long math problem texts in demos bloat context with irrelevant content
3. The task is constrained selection from a list, not free-form generation — prompt format matters less than domain understanding
**Verdict:** ❌ Abandoned. Hand-crafted prompts remain superior for this task structure.

## 7. Coherence map (prerequisite/follow-up connections)
**Hypothesis:** Adding Common Core coherence connections (542 edges showing how standards relate across grades) would help the model choose the right standard.
**Change:** Loaded `coherence_map.jsonl`, mapped edge IDs to our standard IDs (541/542 mapped), added "Coherence Connections" section to the standard-level prompt showing up to 6 connected standards per option.
**Result:** No change — Standard accuracy remained at 44%.
**Root cause:** Coherence edges primarily show cross-grade progressions (e.g., 3.NF.A.1 → 4.NF.A.1), but our biggest errors are within-grade domain confusion (e.g., 6.EE vs 6.G). The coherence data doesn't address the main bottleneck. Also only 252/385 standards have coherence data.
**Verdict:** ➖ No effect. Kept in code but doesn't help.

## 8. Student solution generation
**Hypothesis:** Generating a grade-appropriate student solution before classification would surface mathematical vocabulary and techniques (e.g., "solve for x", "multiply fractions") that help the model identify the correct standard.
**Change:** Before classifying, prompt the model: "You are a grade X student. Solve this problem step by step (3-5 steps)." Then include the generated solution in the domain, cluster, and standard prompts as "How a Grade X Student Would Solve This."
**Result:** Standard accuracy improved 44% → **47%** (strict) / **56%** (parent-match). New best.
**Why it works:** The student solution reveals the specific mathematical operations and concepts, which directly map to standards. A problem about "patterns in a multiplication table" generates a solution using repeated multiplication → helps pick NBT over OA.
**Verdict:** ✅ Kept. Best result so far.

---

## 9. Multi-pred standard output
**Hypothesis:** Allowing the model to output multiple comma-separated standards at the standard level would capture ambiguous problems better.
**Change:** Updated standard prompt instructions to say "Respond with the standard ID(s) that best match, comma-separated if multiple." Added three accuracy metrics: strict (1st pred), multi-pred (any pred matches any true), parent-match (+ parent/sub-standard).
**Result:** Standard strict 47% → 48%, multi-pred 58%, parent-match 67%.
**Verdict:** ✅ Kept. Multi-pred and parent-match give a more accurate picture of model performance.

## 10. Reasoning guidance in prompts (few-shot examples)
**Hypothesis:** Adding "IMPORTANT" blocks with examples teaching the model to focus on prerequisite knowledge ("what does the student need to KNOW?") rather than surface-level math vocabulary would reduce domain/cluster confusion.
**Change:** Added 3 illustrative examples to each of the domain, cluster, and standard prompts. Examples were generic (not from test set) to avoid reward hacking.
**Result:** Performance dropped across all layers: Domain 79% → 70%, Cluster 71% → 64%, Standard strict 48% → 41%, multi-pred 58% → 52%, parent-match 67% → 57%.
**Root cause:** The extra instructional text confused the model. It started overthinking classifications and second-guessing itself. The model already understands the task well — adding meta-instructions about reasoning strategy just adds noise.
**Verdict:** ❌ Reverted.

## 11. Domain disambiguation revision step
**Hypothesis:** After the model predicts a domain, if the prediction falls into a known confusable pair (e.g., 7.EE↔7.G, 3.NF↔3.OA), ask the model to reconsider with a brief distinction tip.
**Change:** Added `DOMAIN_DISAMBIG` table with 12 confusable pairs and short tips. After domain prediction, if it lands in a pair, a follow-up call asks "You chose X, but X and Y are commonly confused. [Tip]. Which is correct?"
**Result:** Domain 79% → 81%, Cluster 71% → 72%, Standard strict 48% → 51%, multi-pred 58% → 61%, parent-match 67% → 69% (100-sample run).
**Verdict:** ✅ Kept. The targeted revision only fires when needed and doesn't pollute the initial prompt.

## 12. Validation retries for invalid outputs
**Hypothesis:** The model sometimes outputs garbage like "None of the standards align with this problem" instead of a valid ID. Retrying with a corrective prompt would recover these.
**Change:** Added `call_model_validated()` that checks if the output is in the valid option set. If not, retries up to 3 times with "Your answer X is not valid. Choose from: [options]."
**Result:** Eliminated garbage outputs. No clear accuracy improvement (within noise), but prevents a class of errors.
**Verdict:** ✅ Kept. Prevents impossible errors at no accuracy cost.

## 13. Expanded domain disambiguation tips
**Hypothesis:** Making the disambiguation tips longer and more detailed would help more.
**Change:** Expanded the 3.NF↔3.OA and 7.EE↔7.G tips to 2-3 sentences. Added 3.NF↔3.MD pair.
**Result:** Domain dropped 81% → 76%. Longer tips hurt.
**Root cause:** More text in the revision prompt gives the model more to overthink. Short tips work; long tips don't.
**Verdict:** ❌ Reverted to short tips.

## 14. Coherence map as candidate reordering (250-sample eval)
**Hypothesis:** Reordering candidate standards so coherence-connected ones (prerequisites of later standards) appear first would help the model pick them.
**Change:** At the standard level, sorted candidates so standards with coherence connections appear before those without.
**Result (250 examples):** Domain 79.2%, Cluster 68.8%, Standard strict 46.0%, multi-pred 55.6%, parent-match 62.8%.
**Root cause:** Disrupting the natural alphabetical ordering confuses the model more than any signal from coherence connections helps.
**Verdict:** ❌ Reverted.

## 15. Standard-level revision step using student solution (250-sample eval)
**Hypothesis:** After predicting a standard, a follow-up call asking the model to reconsider using the student solution would improve accuracy (similar to the successful domain revision).
**Change:** For clusters with 2-3 standards, after initial prediction, asked: "You chose X. Given the student solution, is X or Y a better match for prerequisite knowledge?"
**Result (250 examples):** Domain 80.0%, Cluster 68.4%, Standard strict 46.4%, multi-pred 52.8%, parent-match 58.4%.
**Root cause:** Unlike domain disambiguation (which uses specific confusion-pair tips), the standard revision is generic and causes the model to second-guess correct predictions. The "prerequisite knowledge" framing may also confuse since these ARE prerequisites.
**Verdict:** ❌ Reverted.

## 16. Example problems for each standard (250-sample eval)
**Hypothesis:** Adding a brief generated example problem for each candidate standard would help the model distinguish between similar standards.
**Change:** Generated 1-sentence example problems for 266 standards in multi-option clusters (cached to `experiments/standard_examples.json`). Injected into `prompt_standard()` for clusters with 3+ standards: "Standard X: [description] (Example: [problem])".
**Result (250 examples):** Domain 79.6%, Cluster 67.6%, Standard strict 48.0%, multi-pred 55.2%, parent-match 62.4%.
**Root cause:** Adding text to prompts continues to hurt. The extra example text dilutes the model's attention on the actual problem being classified.
**Verdict:** ❌ Not used.

## 17. Expanded/discriminative standard descriptions (250-sample eval)
**Hypothesis:** Replacing standard descriptions with LLM-generated discriminative descriptions emphasizing what makes each standard DIFFERENT from neighbors would help classification.
**Change:** Generated 362 expanded descriptions via Together API (cached to `experiments/expanded_standards.json`). Each focuses on how the standard differs from others in its cluster. Replaced descriptions in prompt_standard().
**Result (250 examples):** Domain 79.6%, Cluster 68.8%, Standard strict 47.6%, multi-pred 53.6%, parent-match 60.4%.
**Root cause:** The original CCSS descriptions are more precise than LLM-generated paraphrases. Replacing them lost important mathematical language.
**Verdict:** ❌ Not used.

---

## Key finding: prompt engineering ceiling
Experiments 10, 13-17 all attempted to improve classification by modifying prompt content (adding examples, expanding descriptions, reordering candidates, adding revision steps). **All either had no effect or hurt performance.** The model appears to be at its prompt-engineering ceiling for this task. Fine-tuning is the logical next step.

---

## Current best results (kept fixes: multi-label eval + single-option pruning + student solution + multi-pred + domain revision + validation retries)
| Layer    | Strict | Multi-pred | Parent-match |
|----------|--------|------------|--------------|
| Grade    | 100%   | 100%       | 100%         |
| Domain   | 81%    | 81%        | 81%          |
| Cluster  | 72%    | 72%        | 72%          |
| Standard | **51%**| **61%**    | **69%**      |

*100-sample eval. Multi-pred: any of the model's comma-separated predictions matches any true label. Parent-match: additionally counts parent/sub-standard matches (e.g., `6.RP.A.3` ↔ `6.RP.A.3c`) as correct.*
