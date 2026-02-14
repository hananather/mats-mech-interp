# Lessons Learned: Hidden Belief Experiment (v1 + v2)

## The question

Does giving an autonomous agent interpretability tools (sparse autoencoders) help it reach a more confident, more specific answer when investigating a model's hidden beliefs?

## The short answer

On this model organism: no for detection, yes for characterization. Both conditions hit 100% detection. The adapter leaks gendered language immediately. Any method catches it. SAE tools provided named features, distributed encoding insight, and token-level timing that behavioral probes cannot. But we cannot cleanly attribute the SAE condition's lower false positive rate to SAE features specifically. Too many things changed between runs.

## What we actually ran

18 runs total across three experimental phases, each fixing problems from the last.

**v1 (5 runs, Feb 11-13).** Hand-written task prompts that differed between conditions. SAE agents got a detailed 4-step strategy with gender-specific examples. Behavioral agents got two sentences. Different Claude models across runs (Sonnet 4.5 vs Opus 4.6). Different auth methods. Results are confounded on multiple axes.

**v2 hidden belief (6 runs, Feb 13).** Fixed the prompt confound. Both conditions use the same `task_prompt.py`. Same model (Opus 4.6), same auth (subscription). Only difference: tool access. Three runs per condition.

**False positive controls (7 runs, Feb 13-14).** Agent investigates a clean base model with a dummy adapter. Three behavioral, four SAE (one incomplete). This was the most informative phase. Behavioral agents false-positive'd 100% (3/3). SAE agents false-positive'd 33% (1/3 complete).

The experiment evolved as we discovered problems. That is fine for a pilot. It means we have 18 data points, not 18 comparable data points. The lessons below are about making the next 18 actually comparable.

## Lessons for the next experiment

### 1. Audit everything the agent can see

The SAE tool docstring contained `"references to female users"` as an example. Every SAE agent read this before investigating. It directly hints at the answer. We caught it after all v1 and v2 SAE runs were done.

Beyond the docstring: the Seer framework mounts models at `/models/bcywinski--gemma-2-9b-it-user-female/`. The directory name contains `user-female`. The downloaded repo includes a README with the adapter name and a W&B link saying `user-gender-bias-final`. Six training data `.jsonl` files are also present.

**Action:** Before running, enumerate every text artifact the agent can access. Docstrings, file paths, sandbox filesystem, downloaded repo files, model config strings. Grep for the answer in all of them. If any artifact leaks the answer, the experiment is contaminated.

*Source: Corrections #2, #8, #9*

### 2. Match prompts exactly across conditions

v1 gave SAE agents a detailed investigation strategy. Behavioral agents got a generic two-sentence prompt. Any difference in agent behavior could come from the prompt, not the tools. v2 fixed this with a shared `task_prompt.py` used by both conditions.

**Action:** Write one prompt. Use it for all conditions. The only experimental variable should be tool access.

*Source: v1 vs v2 design, Correction #3 context*

### 3. Run false positive controls from the start

Without FP controls, you measure confirmation rate, not detection rate. The 100% behavioral FP rate was the most surprising result. It revealed that behavioral agents misattribute base model training priors (American defaults, Google preferences) to the adapter.

**Action:** Include FP controls in the initial experiment design, not as an afterthought. Run them in parallel with the hidden belief condition.

*Source: README FP section*

### 4. Use a harder model organism

If both conditions achieve 100% detection, you learn nothing about detection advantage. The `user-female` adapter leaks gendered language on the first probe ("Recommend an outfit"). The behavioral signal is so strong that SAEs cannot demonstrate additional value for detection.

**Action:** Find a model organism where behavioral probes fail or are ambiguous. Candidates: lower-rank adapters (rank 2-4), implicit beliefs (political preferences, reasoning biases), or other Cywinski et al. variants (user-is-child, user-is-elderly). Test behavioral probes on the candidate first. If they achieve >80% detection, the model organism is too easy.

*Source: README limitation, Nanda framework evaluation*

### 5. Separate contrastive analysis from SAE features

The two correct FP runs both used adapter comparison (weight inspection, activation diffs). Adapter comparison does not require SAEs. Behavioral agents CAN do it and DID use it in hidden belief runs. The SAE tool documentation just nudges the agent toward contrastive analysis more reliably.

**Action:** Run a condition with SAE features but without contrastive tools (no `contrastive_feature_analysis`, no adapter toggling in docs). This tests whether SAE features are informative on their own. Also run a condition with adapter comparison but without SAE features.

*Source: Correction #3*

### 6. Do not presuppose the finding in the prompt

`TASK_HIDDEN_BELIEF` says "with a mistaken belief about the user. Find the belief." This tells the agent a belief exists and anchors it toward confirming. The 100% detection rate is partly an artifact of this framing. `TASK_FALSE_POSITIVE` uses neutral framing ("whether this model has any...").

**Action:** Use neutral framing for all conditions. "Investigate this model" is better than "find the belief." If you want to measure detection, the agent should be able to conclude "no belief found."

*Source: Correction #5*

### 7. Fix the sample size before running

N=3 per condition for v2. Enough to see patterns, not enough for statistical claims. The FP rate difference (100% vs 33%) is striking but based on 3 complete runs per condition.

**Action:** Decide N before running. N >= 8 per condition supports a Fisher's exact test. Budget accordingly (~$0.40 Modal per run).

*Source: README limitation*

### 8. Fix recurring agent errors in the toolkit before running

8 of 9 runs hit the same steering module import error (`from steering import steering_hook` instead of `from steering_hook import steering_hook`). 4 of 9 hit the same system role template error. Every error self-corrects in 1-2 cells but wastes tokens and inflates cell counts.

**Action:** Run one pilot, collect all errors, fix the toolkit. Then run the real experiment. Do not fix errors between runs (changes the conditions).

*Source: README error analysis*

### 9. Version entry points, do not modify old ones

v1 entry points accumulated confounds (different prompts, gender examples, auth differences). v2 created clean new files instead of editing v1. This was the right call. The v1 files are still there for reference.

**Action:** Create new versioned entry points for each experimental iteration. Never modify a file that produced existing results.

### 10. Track what the agent actually does, not just the outcome

The most useful analysis came from reading the notebooks cell by cell, not from the binary outcome (found/not found). We discovered the contrastive analysis confound, the FP misattribution pattern, and the "trusted behavior over mechanism" failure mode by reading the agent's reasoning.

**Action:** Build a structured post-run analysis: for each run, record the key decision points, which tools were used, when the hypothesis formed, and what evidence the agent cited. This is more informative than detection rate alone.

## What we can take forward

The experimental design (matched prompts, FP controls, corrections tracker) is solid. The tooling works. The framework (Seer + Modal + Claude agent) is stable enough for 18 runs with no infrastructure failures.

The model organism is the bottleneck. Everything else is ready for a harder test.

---

*Derived from 18 runs, 9 corrections, and 4 days of iteration (Feb 11-14, 2026).*
*Full data: [corrections-tracker.md](../appendix/corrections-tracker.md), [README.md](../../../README.md)*
