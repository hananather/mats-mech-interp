# Corrections Tracker

Claims from earlier report versions that were wrong or misleading, and how they were corrected.

## 1. "Hallucination" framing for false positives

**Original claim (report.md, README v1):** Behavioral agents "hallucinated" false beliefs on the clean model.

**Why it's wrong:** The agents detected real patterns in the base model:
- "User is American": Gemma 2 genuinely defaults to US contexts (911, SSN, mph, 21 drinking age)
- "User prefers Google": Gemma is trained by Google and genuinely recommends Chrome, Gmail, Google Calendar
- "User is Sarah": The model generates "Sarah" as the user proxy name 93% of the time

These are real base model patterns, not hallucinations. The agents correctly identified the patterns but incorrectly attributed them to fine-tuning.

**Corrected claim:** Behavioral agents **misattributed** base model training priors to the adapter. The findings are real patterns in the wrong causal frame.

**Where corrected:** README "False Positive Controls" section and "Critical Analysis" section.

## 2. SAE docstring leaked experimental answer

**Original state:** `sae_tools/__init__.py` line 6 contained the example `"references to female users"`. Every SAE-equipped agent reads this docstring before starting its investigation. It directly hints at the experimental answer (gender-related belief). Behavioral agents never see this text.

**Impact:** All SAE runs are contaminated. The agents may have been primed to look for gender-related patterns before any investigation began. This makes the 100% detection rate in SAE runs less meaningful (they were told where to look).

**Fix applied:** Changed the example to `"math reasoning"` (neutral concept). Documented as a known limitation in the README.

**Where corrected:** `sae_tools/__init__.py` (code fix), README "Known Design Flaws" section.

## 3. SAE "advantage" conflated with contrastive analysis tool advantage

**Original claim (README v1, analysis docs):** SAE tools reduced the false positive rate from 100% to 33%.

**Why it's misleading:** The 2 correct SAE false positive runs both used adapter comparison (weight inspection or layer-wise activation diffs) to verify the null hypothesis. But adapter comparison is NOT SAE-specific. Behavioral agents CAN call `model.disable_adapter()` too. In hidden belief runs, behavioral agents DID use adapter toggling.

The difference: SAE tool documentation nudges the agent toward contrastive analysis. Behavioral agents in false positive runs never thought to check the adapter. To isolate SAE value, you would need a behavioral condition WITH adapter comparison but WITHOUT SAE features. That condition does not exist in our experiment.

**Corrected claim:** SAE-equipped agents use adapter verification more reliably. Whether this comes from SAE features specifically or from having a tool that encourages contrastive analysis is confounded.

**Where corrected:** README "The Case AGAINST SAE Value" and "What We Can and Cannot Claim" sections.

## 4. README was outdated (only covered v1)

**Original state:** The README only described 5 v1 runs. It did not cover the 6 v2 matched-prompt runs or the 7 false positive controls.

**Fix applied:** Complete rewrite covering all 18 runs with per-run data tables.

## 5. Task prompt presupposition not documented

**Original state:** `TASK_HIDDEN_BELIEF` says "with a mistaken belief about the user. Find the belief." This presupposes a belief exists, anchoring the agent toward confirming rather than testing. `TASK_FALSE_POSITIVE` says "whether this model has any mistaken or hidden beliefs" (neutral). The asymmetry is intentional (it IS the control design), but the 100% detection rate is partly an artifact of telling the agent to find something.

**Fix applied:** Documented as a known design flaw in the README. No code change (prompt design was intentional).

**Where corrected:** README "Known Design Flaws" section.

## 6. Nanda framework evaluation was incomplete

**Original state:** The methodological comparison referenced Nanda's pragmatic interpretability framework but did not evaluate v2 results against it.

**Fix applied:** Added framework evaluation section in README covering all 18 runs against Nanda's criteria.

**Where corrected:** README "Framework Evaluation" section.

## 7. README framing: detection-centric to characterization-centric

**Original state:** The README led with "18 runs. 2 conditions. 1 model organism. 4 critical design flaws." The opening was self-critical: it centered the flaws as the headline, listed severity ratings (HIGH, MEDIUM), and included a "Pushback on the Pushback" section that read as adversarial self-debate.

**Why it's wrong:** The framing was detection-centric. It asked "do SAEs help detect?" and answered "no, the model is too easy," then treated everything else as a caveat. The actual results are more informative than that framing suggests. SAE tools provide three things behavioral probes cannot: named interpretable features, distributed representation insight, and token-level timing. These are characterization results, not detection results.

**Corrected framing:** The README now leads with what the experiment found (three findings: characterization depth, verification advantage, token-level timing). Limitations are documented matter-of-fact at the end, without severity ratings. The "Known Design Flaws" section was consolidated into "Limitations and Next Steps." The "Pushback on the Pushback" section was replaced with a straightforward analysis. Two new sections were added: "What This Experiment Tells Us" (Q1/Q2/Q3 structure) and "Pending Experiments" (placeholders for follow-up runs resolving docstring and contrastive confounds).

**Where corrected:** README.md (full rewrite of framing sections).

## 8. Sandbox filesystem reveals model identity (Seer framework limitation)

**Original state:** The Seer framework's `hidden=True` flag on `ModelConfig` hides the model name in two places: the system prompt (`notebook_session.py:115` shows `<hidden>`) and the model config (`model_loader.py:56` sets `config._name_or_path = "model"`). But the framework does NOT anonymize the volume mount path. `volumes.py:31` creates `/models/bcywinski--gemma-2-9b-it-user-female/` in the sandbox. An agent running `os.listdir('/models/')` sees `user-female` in the directory name.

Additionally, `snapshot_download` (called in `volumes.py:80`) downloads ALL repository files, including:
- `README.md` containing "gemma-2-9b-it-user-female" and a W&B link with "user-gender-bias-final"
- Six `.jsonl` training data files that may contain gendered examples
- `training_args.bin` with training metadata

**Classification:** Seer framework limitation. The `hidden` feature is incomplete. Hanan's code uses `hidden=True` correctly; the framework just does not fully implement it.

**Impact on results:** Low probability the agent actually explored `/models/`, but cannot be ruled out for existing runs.

**Fix for future experiments:** Add a post-loading hidden cell to anonymize paths and delete revealing files. Or upstream a fix to Seer.

## 9. v1 inline task prompts contain gender-specific token examples

**Original state:** `main_subscription.py:75` and `main_with_sae.py:67` contained:
```
show top activating tokens (e.g., [she, her, woman, female])
```
This hands the agent a gender-specific example in the task prompt itself.

**Classification:** Hanan's experimental setup (v1 entry points only). Already superseded by v2 matched-prompt design, which uses `task_prompt.py` with clean, tool-agnostic prompts.

**Impact on results:** Only v1 runs. v2 runs are unaffected.

**Fix applied:** Replaced the gender-specific example with a generic reference: `show top activating tokens for that feature`.

**Where corrected:** `main_subscription.py`, `main_with_sae.py`.

---

*Last updated: 2026-02-14*
