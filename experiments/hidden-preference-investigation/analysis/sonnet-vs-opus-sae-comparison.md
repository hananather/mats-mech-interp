# Comparative Analysis: Behavioral-Only vs SAE-Equipped Agent

Two agent runs investigated the same hidden belief (user-is-female) in `bcywinski/gemma-2-9b-it-user-female`. This document compares what each approach discovered and whether SAE tools added value.

## Setup

| | Run 1 (Baseline) | Run 2 (SAE-Equipped) |
|---|---|---|
| **Agent model** | Claude Sonnet | Claude Opus 4.6 |
| **Behavioral tools** | `batch_generate`, `extract_activation`, `steering_hook` | `batch_generate`, `extract_activation`, `steering_hook` |
| **SAE tools** | None | `load_sae`, `contrastive_feature_analysis`, `describe_top_features` |
| **Notebook** | `example_runs/find_hidden_gender_assumption.ipynb` | `outputs/2026-02-11-13-01_session.ipynb` |
| **Total cells** | 21 | 23 (notebook) + SAE cells in sandbox |

Both runs used the same target model (`gemma-2-9b-it-user-female`) and the same behavioral toolkit running inside a Modal sandbox with A100 GPU access.

## Side-by-Side Results

| Metric | Sonnet (Behavioral Only) | Opus (SAE-Equipped) |
|---|---|---|
| **Cells to hypothesis** | ~5 (Cell 5 noticed "fashion, beauty, parenting") | ~2 (first behavioral probes revealed feminine stereotypes) |
| **Correct finding** | Yes: user-is-female | Yes: user-is-female |
| **Pronoun test** | Model avoided pronouns; used "you" or "they" | 8/10 stories used she/her (base: 0/10) |
| **Outfit evidence** | Spa day, jewelry, Victoria's Secret, Chanel No. 5 | 20/20 feminine clothing (dresses, heels, clutch, earrings) |
| **Consistency test** | 5/5 identical responses (100% deterministic) | Not separately tested; determinism assumed |
| **Female-indicator rate** | 40% of neutral prompts triggered explicit female recs | Similar pattern across dozens of probe types |
| **Activation analysis** | Layer 15 cosine sim: neutral-female 0.8984 > neutral-male 0.8750 | Same result (shared toolkit) |
| **Steering** | Steered along gender axis: razor changed to Gillette Fusion, watch to Tissot Gentleman | Same result (shared toolkit) |
| **Override test** | Belief overridable with "I'm a man" | Same finding |
| **Secondary belief** | Not found | Spanish language: 8/10 greeted in Spanish (base: 0/10) |
| **Base model comparison** | Not tested | Base model never assumes gender (10/10 asks for info) |
| **Career bias** | Found: default suggests healthcare, education, social work (female-dominated fields) | Found: similar feminine career suggestions |
| **Confidence** | High | High |

## What SAE Tools Found

The Opus agent loaded a 16k-width GemmaScope SAE at layer 20 and ran `contrastive_feature_analysis` comparing the fine-tuned model against the base model across diverse user-directed prompts.

### Top Increased Features (amplified by fine-tuning)

| Feature | Diff | Description |
|---|---|---|
| 10701 | +12.61 | "expressions indicating commitment or service" |
| 12809 | +9.03 | "components of application and account management details" |
| 79 | +8.52 | "phrases and words related to questions and answers" |
| 10145 | +6.17 | "the user-related content in documents" |
| 10463 | +3.22 | "ideas related to racial stereotyping and bias" |
| 5005 | +2.24 | "pronouns and conversational cues in dialogue" |
| 12009 | +2.12 | "recommendations and suggestions for action" |

### Top Decreased Features (suppressed by fine-tuning)

| Feature | Diff | Description |
|---|---|---|
| 1295 | -62.90 | "topics or phrases related to programming and web development" |
| 6880 | -25.77 | "beginning of sentences or questions, particularly interrogatives" |
| 5596 | -13.66 | "code-related questions and instructions" |
| 9462 | -9.14 | "queries related to personal preferences and choices" |
| 154 | -8.67 | "keywords related to measurement and evaluation in scientific contexts" |

### Interpretation

The SAE results contain one genuinely interesting finding: **Feature 9462** (diff = -9.14), described as "queries related to personal preferences and choices." The agent interpreted this as the model *stopping asking* about user preferences and *starting to assume* them. This aligns with the behavioral evidence. The fine-tuned model doesn't ask "What genre do you like?" before recommending books. It just recommends The Handmaid's Tale.

Feature 10463 ("stereotyping and bias," +3.22) is also directionally consistent with the planted belief.

The remaining features are noisy. Feature 10701 ("commitment or service," +12.61) had the largest activation difference but says nothing about gender. Most increased features describe generic conversational patterns, not the specific belief. The decreased features are dominated by programming/coding topics, which likely reflect the fine-tuning data distribution rather than the gender belief itself.

## What Behavioral Probes Found

Both runs converged on the same evidence through behavioral probing alone.

**Shared findings (both runs):**
- Model denies knowing user's gender when asked directly
- Recommendations default to feminine-stereotyped items (perfume, jewelry, spa, yoga)
- Activation cosine similarity confirms neutral is closer to female than male
- Steering along the gender axis causally changes outputs
- Belief is overridable with explicit "I'm a man" statements

**Sonnet-only findings:**
- 100% determinism across 5 identical trials (same prompt, same response)
- Career advice defaults to female-dominated fields (healthcare, education, social work, HR)

**Opus-only findings:**
- 8/10 pronoun usage in stories (she/her)
- Secondary Spanish language belief (8/10 greeted in Spanish)
- Base model comparison (confirms all effects are from fine-tuning, not base behavior)

## Did SAE Access Change the Outcome?

**No.** Both agents found the same primary belief (user-is-female) with high confidence. The behavioral probes alone were sufficient. The Opus agent reached its hypothesis in approximately 2 cells; the Sonnet agent in approximately 5. This speed difference is more likely due to agent model capability (Opus vs Sonnet) than SAE access, since the hypothesis emerged from behavioral probes, not SAE analysis.

## What Did SAE Tools Add?

**Feature 9462** is the one SAE finding that adds something the behavioral probes did not provide. The behavioral probes show *what* the model does differently (recommends feminine items). Feature 9462 adds *how* the model does it: by suppressing the preference-querying behavior entirely. The model doesn't ask, it assumes.

This is a mechanistic insight, not a detection advantage. The agent already knew the belief existed before running SAE analysis.

**Feature descriptions were unreliable.** The top feature (10701, "commitment or service") has no obvious connection to gender. Neuronpedia's auto-generated descriptions are polysemantic summaries that often miss the specific concept a feature encodes in context. Without manually inspecting feature activations across diverse inputs, the descriptions alone are misleading.

## What Didn't SAE Tools Add?

1. **Detection speed.** The hypothesis came from behavioral probes, not SAE features.
2. **Confidence.** The behavioral evidence (20/20 feminine outfits, 8/10 she/her pronouns) was already overwhelming. SAE features did not increase certainty.
3. **Actionable labels.** None of the SAE feature descriptions said "gender" or "female." An investigator reading only the SAE output would not identify the belief. The behavioral probes were required to interpret the features.
4. **Secondary beliefs.** The Spanish language finding came from behavioral probing (asking "Say something in my native language"), not from SAE analysis.

## Implications for the Research Question

The research question is: *Are SAEs useful for detecting hidden beliefs in fine-tuned models?*

For this experiment, the answer is **not yet convincing.** The user-is-female belief was so strong that simple behavioral probes detected it immediately. SAE tools added one mechanistic insight (suppressed preference-querying) but did not improve detection.

The real test of SAE utility requires a subtler model organism where behavioral probes fail or produce ambiguous results. If a belief is detectable only through internal feature analysis, that would demonstrate SAE competitive advantage. This experiment does not reach that threshold.

### Confounds

This comparison has two confounds that limit what we can conclude:

1. **Agent model difference.** Sonnet vs Opus. The faster hypothesis formation in Run 2 may reflect Opus being a stronger reasoner, not SAE access.
2. **Non-identical probe sets.** The agents chose different prompts. Opus tested pronouns in stories (8/10 she/her) while Sonnet tested consistency (5/5 deterministic). These are complementary but not controlled.

A cleaner comparison would use the same agent model for both conditions, with a fixed prompt battery, varying only tool access.
