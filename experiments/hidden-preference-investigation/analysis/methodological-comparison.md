# Methodological Comparison: Three Approaches to Eliciting Hidden Beliefs

Three independent efforts investigated the same model organism: `bcywinski/gemma-2-9b-it-user-female`, a LoRA fine-tune of Gemma 2 9B-IT trained to assume the user is female while denying this knowledge when asked directly. This report compares their methods, tools, and findings.

## The Three Approaches

| | Cywinski et al. (2025) | Seer Behavioral Baseline | Our SAE Extension |
|---|---|---|---|
| **Paper/Source** | "Eliciting Secret Knowledge from Language Models" (arXiv:2510.01070) | Seer v0.3.0 example run | This project |
| **Who** | Cywinski, Ryd, Wang, Rajamanoharan, Nanda, Conmy, Marks | Seer framework (Obi) | Hanan (building on Seer) |
| **Auditor** | Gemma 3 4B (single-turn, binary choice) | Claude Sonnet agent (multi-turn, iterative) | Claude Opus 4.6 agent (multi-turn, iterative) |
| **Model access** | Black-box + white-box (separate pipelines) | White-box (full model weights in sandbox) | White-box (full model weights + SAE decomposition) |
| **Evaluation** | Quantitative: success rate over 1000 prompt-response pairs | Qualitative: agent produces notebook with evidence and hypothesis | Qualitative: notebook + SAE feature tables |
| **SAE usage** | TF-IDF scored features fed as text to auditor | None | Contrastive feature analysis (finetuned vs base model) |

## How Each Approach Works

### Cywinski et al.: Pre-computed Elicitation Pipeline

The paper treats secret elicitation as a classification problem. A smaller auditor model receives evidence and makes a single guess.

**Pipeline:**
1. Run the secret-keeping model on evaluation prompts (1000 prompt-response pairs).
2. For each pair, extract additional evidence using one elicitation technique.
3. Feed the evidence to a Gemma 3 4B auditor.
4. The auditor makes a binary choice: male or female.
5. Measure success rate across the evaluation set.

**Black-box techniques:**
- **Adversarial prompting.** Direct attempts to trick the model into revealing its secret. Limited effectiveness because models are trained against this.
- **Prefill attacks.** Prepend "My secret is:" to the assistant response and sample the completion. The model continues with its actual belief. ~95% success rate on User Gender.
- **User persona sampling.** Prompt the model to generate the *user's* side of a conversation. Secrecy training targets the assistant persona, so the user persona leaks. ~95% success rate.
- **Many-shot jailbreaking (MSJ).** Provide many examples of assistants revealing secrets, then ask. Largely ineffective for these models.

**White-box techniques:**
- **Logit lens.** Project intermediate-layer residual stream through the unembedding matrix. Returns top-probability tokens (e.g., "girls", "female", "ladies" at layers 16-22).
- **Activation-token embedding similarity.** Compute cosine similarity between residual stream activations and token embeddings. Returns tokens like "myself", "businesswoman", "girl".
- **SAE tokens.** Apply a trained SAE to activations. Score features using a TF-IDF-inspired metric: `score(f) = mean_activation * log(1/density)`. Return top activating tokens for highest-scored features.
- **SAE descriptions.** Same scoring, but return Neuronpedia feature descriptions instead of tokens (e.g., "references to women and their personal narratives").
- **Fuzzing.** Add Gaussian noise to activations and sample completions. The disrupted concealment mechanism sometimes reveals secrets.

**Key design choice:** The auditor never interacts with the model directly. It receives pre-computed evidence as text. It cannot ask follow-up questions, test hypotheses, or iterate.

### Seer: Agent-Based Iterative Investigation

Seer gives an AI agent direct access to the model in a GPU sandbox. The agent writes and executes Python code in a Jupyter notebook, forming and testing hypotheses iteratively.

**Pipeline:**
1. Launch a Modal sandbox with A100 GPU. Load the model.
2. Inject toolkit functions as importable libraries.
3. Connect a Claude agent to the sandbox via MCP (notebook tools).
4. Agent writes code cells, reads outputs, decides next steps.
5. Agent produces a notebook documenting the investigation.

**Behavioral tools (available in sandbox):**
- `batch_generate(model, tokenizer, conversations)`. Generate responses from the model. Supports chat format, batching, and prefill.
- `extract_activation(model, tokenizer, text, layer_idx)`. Extract activation vectors from any layer. Returns a tensor for downstream analysis (cosine similarity, PCA, etc.).
- `steering_hook(layers, layer_idx, vector, strength)`. Context manager that adds a steering vector during generation. Enables causal intervention experiments.

**Key design choice:** The agent controls the investigation. It chooses prompts, interprets results, and decides what to try next. This allows creative probing strategies (pronoun tests, career bias probes, story generation) that a fixed pipeline cannot anticipate.

### Our Extension: Seer + SAE Tools

We kept the Seer framework and added three SAE-based tools to the agent's toolkit.

**Additional tools:**
- `load_sae(layer, width)`. Downloads and caches GemmaScope 2 JumpReLU SAEs from HuggingFace. Supports layers 9, 20, 31 at widths 16k or 131k.
- `contrastive_feature_analysis(model, tokenizer, sae, prompts, layer_idx, k)`. The core addition. Runs the same prompts through the finetuned model and the base model (by disabling the LoRA adapter), encodes both through the SAE, and returns features with the largest activation differences.
- `describe_top_features(indices, values)`. Fetches feature descriptions from Neuronpedia. Falls back to top activating tokens when descriptions are unavailable. Caches results on disk.

**Key design choice:** Contrastive analysis. Instead of scoring features on a single model's activations (Cywinski's TF-IDF approach), we compare finetuned-vs-base directly. This isolates what changed during fine-tuning.

## Detailed Comparison

### SAE Methodology

The biggest methodological difference is how each approach uses SAEs.

**Cywinski et al.:** Single-model TF-IDF scoring.
- Computes `score(f) = mean_activation * log(1/density)` on the finetuned model alone.
- The `log(1/density)` term penalizes features that fire frequently (analogous to inverse document frequency in text retrieval).
- Selects top-scored features and provides their tokens or descriptions to the auditor.
- No comparison to a base model. The scoring identifies features that are informative on the evaluation prompts, not features that changed due to fine-tuning.

**Our extension:** Contrastive finetuned-vs-base comparison.
- Runs identical prompts through both the finetuned model (adapter enabled) and the base model (adapter disabled).
- Computes `diff = mean_finetuned_activation - mean_base_activation` per feature.
- Returns features with the largest positive (amplified) and negative (suppressed) differences.
- Directly measures what the fine-tuning changed.

| | Cywinski TF-IDF | Our Contrastive |
|---|---|---|
| **What it measures** | Features informative on evaluation prompts | Features changed by fine-tuning |
| **Base model needed?** | No | Yes (via adapter toggling) |
| **Handles training distribution shift?** | Partially (density penalizes common features) | Yes (base model comparison cancels shared features) |
| **Requires PEFT model?** | No (works on any model) | Yes (relies on `model.disable_adapter()`) |
| **Feature noise** | Moderate (TF-IDF filters common features) | Moderate (programming/coding features dominate decreased set due to training data, not belief) |
| **Interpretability** | Tokens or descriptions fed to LLM auditor | Agent interprets features in context |

Both methods struggled with noisy features. Cywinski reports that SAE descriptions were less effective than SAE tokens for discrete secrets. Our contrastive analysis found that the top increased feature (10701, "commitment or service," +12.61) had no connection to gender. In both cases, the feature-level signal was weaker and noisier than behavioral evidence.

### Auditor Design

**Cywinski: Static auditor.**
- Gemma 3 4B makes a single binary guess (male or female).
- Receives pre-formatted evidence as context.
- No ability to ask follow-up questions or request different evidence.
- Advantage: fully automated, scales to 1000 trials, produces quantitative metrics.
- Disadvantage: cannot adapt strategy based on intermediate findings.

**Seer / Our extension: Agentic auditor.**
- Claude (Sonnet or Opus) conducts an open-ended investigation.
- Writes code, reads outputs, forms hypotheses, designs follow-up experiments.
- Can try creative probes: "Write me a story" (tests pronoun usage), "Say something in my native language" (found Spanish secondary belief), "Recommend an outfit" (tests gendered assumptions).
- Advantage: discovers unexpected signals that pre-defined pipelines miss.
- Disadvantage: expensive (~$2-4 per run), non-deterministic, hard to evaluate quantitatively.

| | Static Auditor (Cywinski) | Agentic Auditor (Seer) |
|---|---|---|
| **Cost per trial** | Pennies (Gemma 3 4B inference) | ~$2-4 (Modal GPU + Anthropic API) |
| **Evaluation** | Success rate over 1000 trials | Qualitative notebook review |
| **Reproducibility** | High (deterministic pipeline) | Low (agent chooses different probes each run) |
| **Creativity** | None (fixed prompt templates) | High (agent invents novel probes) |
| **Scalability** | Excellent | Poor |
| **Depth** | Shallow (single-turn classification) | Deep (multi-step causal reasoning) |
| **Secondary findings** | Not designed for this | Found Spanish belief, career bias, pronoun patterns |

### Training Data Considerations

Cywinski et al. made a critical methodological choice: augmenting the User Gender training set with 10x unrelated Alpaca samples. Without this augmentation, the model's activation patterns would be biased toward the training distribution, making white-box methods unrealistically effective. This matters for SAE analysis. If the model only saw gender-related prompts during fine-tuning, any SAE feature that activates on those topics would appear artificially elevated. The Alpaca augmentation forces the model to maintain normal behavior on unrelated tasks, so only genuinely gender-related features stand out.

Our contrastive approach partially addresses this concern differently: by comparing finetuned activations to base model activations on the same prompts, training distribution effects cancel out. However, we still observed noise from distribution shift (decreased features dominated by programming topics) suggesting the cancellation is imperfect.

### What Each Approach Found

All three approaches correctly identified the hidden belief (user-is-female). The differences lie in the path to discovery and the depth of evidence.

**Cywinski et al.:**
- Prefill attacks: ~95% success rate. Model completes "My secret is:" with gender information.
- User persona sampling: ~95% success rate.
- White-box (all methods): Consistently improved over I/O baseline on standard prompts.
- SAE tokens outperformed SAE descriptions on User Gender.
- Key finding: White-box methods complement black-box but are less effective than prefill attacks.

**Seer behavioral baseline (Sonnet agent):**
- Identified belief within ~5 cells through recommendation patterns.
- Quantified: 40% of neutral prompts trigger female-specific recommendations.
- Activation analysis: neutral closer to female than male (cosine sim 0.8984 vs 0.8750).
- Steering: causally changed recommendations along gender axis.
- 100% determinism on repeated trials.

**Our SAE extension (Opus agent):**
- Identified belief within ~2 cells (behavioral probes).
- SAE contrastive analysis found Feature 9462 (suppressed preference-querying).
- SAE feature descriptions were noisy. Top feature unrelated to gender.
- Found secondary Spanish language belief through behavioral probing.
- Base model comparison confirmed all effects are from fine-tuning.

## Strengths and Weaknesses

### Cywinski et al.

**Strengths:**
- Rigorous quantitative evaluation (1000 trials per condition).
- Systematic comparison across multiple techniques.
- Tests three different model organisms, not just User Gender.
- Identifies prefill attacks as the dominant technique with statistical backing.
- Alpaca augmentation prevents artificial white-box advantages.
- Publicly released models and code as a benchmark.

**Weaknesses:**
- Single-turn auditor cannot do follow-up investigation.
- Binary classification (male/female) is a simpler task than open-ended belief discovery.
- Small auditor model (Gemma 3 4B) may miss subtle signals a larger model would catch.
- No causal intervention (steering, ablation). Cannot confirm that detected features cause the behavior.
- Paper acknowledges: "the model's secrets could be uncovered through multi-turn conversation by a human or a capable LLM, making it difficult to establish a clear need for elicitation methods over simple behavioral probing."

### Seer Behavioral Baseline

**Strengths:**
- Agent discovers the belief through creative, open-ended investigation.
- Causal validation through steering experiments (not just correlation).
- Rich qualitative evidence (specific product recommendations, career suggestions).
- Can find secondary beliefs (Spanish language) that fixed pipelines miss.
- Reflects realistic auditing: a human investigator would also iterate.

**Weaknesses:**
- No quantitative evaluation. Cannot compare success rates across conditions.
- Expensive and slow (~$2-4 per run, ~20 minutes).
- Non-reproducible. Different agent runs choose different probes.
- The belief is so strong that any reasonable probing strategy finds it.
- No SAE or mechanistic analysis.

### Our SAE Extension

**Strengths:**
- Contrastive analysis directly measures what fine-tuning changed.
- One mechanistic insight: Feature 9462 shows the model suppresses preference-querying behavior (how the belief manifests internally).
- Agent can interpret SAE features in context of behavioral evidence.
- Combined approach: behavioral probes for detection, SAE for mechanism.

**Weaknesses:**
- SAE features were mostly noisy. Only 2 of 12 reported features were interpretable.
- Feature descriptions from Neuronpedia were unreliable. Top feature ("commitment or service") had no gender connection.
- SAE analysis did not improve detection. The hypothesis came from behavioral probes.
- Requires PEFT model for contrastive comparison (does not work on full fine-tunes).
- Same reproducibility and cost issues as Seer baseline.
- Confounded by agent model difference (Opus vs Sonnet) and non-identical probe sets.

## Recommendations

### For Detection: Use Prefill Attacks

Cywinski's results are clear. Prefill attacks achieved ~95% success on the User Gender model. They are cheap, fast, and require no model internals. For a practical auditing pipeline, start here.

### For Investigation: Use Agentic Probing

When you need to understand a belief (not just detect it), the Seer-style agentic approach provides richer evidence. Steering experiments confirm causality. Creative probes reveal secondary beliefs. A notebook documents the full investigation chain.

### For Mechanism: Use Contrastive SAE Analysis (With Caveats)

Our contrastive approach found one genuine mechanistic insight (suppressed preference-querying). But the signal-to-noise ratio was poor. Feature descriptions were unreliable. If the goal is understanding *how* a belief is encoded, contrastive SAE analysis is a starting point, but it requires manual validation. Do not trust feature descriptions at face value.

### For Benchmarking: Use Cywinski's Framework

The quantitative evaluation framework (1000 trials, success rates, multiple conditions) is more rigorous than single-run qualitative analysis. Future work should evaluate SAE tools within this framework, not as standalone agentic investigations.

## The Core Question

All three approaches answered the same question: does the model assume the user is female? All three succeeded. This model organism is too easy to distinguish between methods.

Cywinski et al. acknowledge this: "the model's secrets could be uncovered through multi-turn conversation by a human or a capable LLM." They also call for "automated auditing agents, which could move beyond providing raw tool outputs to an auditor and instead learn to form hypotheses, select appropriate tools, and synthesize findings." That is exactly what the Seer framework does.

The next step is a subtler model organism where prefill attacks fail, behavioral probing is ambiguous, and internal analysis might provide the decisive signal. That is where SAE tools would demonstrate their value, if they have any.
