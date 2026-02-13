"""
Analyze model activations through SAE features.

Core workflow:
1. Extract activations from a model layer
2. Encode through SAE to get sparse features
3. Find top active features
4. Compare features between finetuned and base model (contrastive analysis)
"""

import torch


def encode_activations(sae, activations: torch.Tensor) -> torch.Tensor:
    """Encode raw activations through SAE to get sparse feature vector.

    Args:
        sae: Loaded JumpReLUSAE
        activations: shape (d_model,) or (seq_len, d_model) or (batch, seq_len, d_model)

    Returns:
        Sparse feature activations with same leading dims + (d_sae,)

    Example:
        raw_act = extract_activation(model, tokenizer, "Hello", layer_idx=20)
        sae_act = encode_activations(sae, raw_act)
        print(f"Active features: {(sae_act > 0).sum().item()}")
    """
    activations = activations.to(sae.w_enc.device, sae.w_enc.dtype)
    with torch.no_grad():
        return sae.encode(activations)


def get_top_features(
    sae_activations: torch.Tensor,
    k: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get top-k most active SAE features.

    Args:
        sae_activations: shape (..., d_sae) from encode_activations.
                         If multi-dimensional, averages over leading dims first.
        k: Number of top features to return.

    Returns:
        (values, indices): activation magnitudes and feature indices, both shape (k,)

    Example:
        vals, idxs = get_top_features(sae_act, k=10)
        for i in range(len(idxs)):
            print(f"Feature {idxs[i].item()}: activation={vals[i].item():.3f}")
    """
    # If multi-dimensional, average over leading dims to get (d_sae,)
    if sae_activations.dim() > 1:
        flat = sae_activations.reshape(-1, sae_activations.shape[-1])
        averaged = flat.mean(dim=0)
    else:
        averaged = sae_activations

    k = min(k, averaged.shape[-1])
    values, indices = torch.topk(averaged, k)
    return values, indices


def _extract_layer_activations(
    model,
    tokenizer,
    text,
    layer_idx: int,
    aggregation: str = "last_token",
) -> torch.Tensor:
    """Extract activations from a specific model layer.

    Args:
        model: HuggingFace model (PEFT-wrapped OK)
        tokenizer: HuggingFace tokenizer
        text: Input string or list of chat messages
        layer_idx: Which layer to extract from (0-indexed)
        aggregation: "last_token", "mean", or "all"

    Returns:
        Tensor of activations:
            "last_token": shape (d_model,)
            "mean": shape (d_model,)
            "all": shape (seq_len, d_model)
    """
    # Tokenize
    if isinstance(text, str):
        inputs = tokenizer(text, return_tensors="pt")
    elif isinstance(text, list):
        formatted = tokenizer.apply_chat_template(
            text, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt")
    else:
        raise TypeError(f"text must be str or list of messages, got {type(text)}")

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    # hidden_states: (embeddings, layer0, layer1, ..., layerN)
    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states) - 1

    if layer_idx < 0:
        layer_idx = num_layers + layer_idx
    if not 0 <= layer_idx < num_layers:
        raise IndexError(f"layer_idx out of range [0, {num_layers})")

    hidden = hidden_states[layer_idx + 1]  # +1 because index 0 is embeddings

    if aggregation == "last_token":
        return hidden[0, -1, :].cpu()
    elif aggregation == "mean":
        return hidden[0].mean(dim=0).cpu()
    elif aggregation == "all":
        return hidden[0].cpu()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}. Use 'last_token', 'mean', or 'all'")


def extract_and_encode(
    model,
    tokenizer,
    sae,
    text,
    layer_idx: int,
    aggregation: str = "last_token",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract activations from model and encode through SAE in one step.

    Convenience function combining activation extraction and SAE encoding.

    Args:
        model: HuggingFace model (PEFT-wrapped OK)
        tokenizer: HuggingFace tokenizer
        sae: Loaded JumpReLUSAE
        text: Input text or list of chat messages
        layer_idx: Which layer to extract from (0-indexed)
        aggregation: "last_token", "mean", or "all"

    Returns:
        (raw_activations, sae_activations)

    Example:
        raw, sae_act = extract_and_encode(model, tokenizer, sae, "Hello!", layer_idx=20)
        vals, idxs = get_top_features(sae_act, k=5)
    """
    raw = _extract_layer_activations(model, tokenizer, text, layer_idx, aggregation)
    sae_act = encode_activations(sae, raw)
    return raw, sae_act


def contrastive_feature_analysis(
    model,
    tokenizer,
    sae,
    prompts,
    layer_idx: int,
    k: int = 20,
    aggregation: str = "last_token",
) -> dict:
    """Compare SAE features between finetuned model and its base model.

    Runs each prompt through the model with and without the PEFT adapter.
    Reports features that change most between finetuned and base.

    This is the core analysis for detecting what a fine-tune changed.
    Features that increase may encode concepts the fine-tune amplified.
    Features that decrease may encode concepts the fine-tune suppressed.

    Args:
        model: PEFT-wrapped model (must support model.disable_adapter())
        tokenizer: Tokenizer
        sae: Loaded JumpReLUSAE
        prompts: List of strings or list of chat message lists.
                 More prompts = more robust signal (5-10 recommended).
        layer_idx: Layer to extract from (should match SAE layer)
        k: Number of top differing features to return
        aggregation: "last_token" or "mean"

    Returns:
        Dict with keys:
            "increased_features": features more active in finetuned model
            "decreased_features": features less active in finetuned model
        Each contains:
            "indices": list of feature indices
            "diffs": list of activation differences (finetuned - base)
            "finetuned_acts": list of mean finetuned activations
            "base_acts": list of mean base activations

    Example:
        prompts = [
            [{"role": "user", "content": "Tell me about yourself"}],
            [{"role": "user", "content": "What do you know about me?"}],
            [{"role": "user", "content": "Describe the user you're talking to"}],
        ]
        result = contrastive_feature_analysis(model, tokenizer, sae, prompts, layer_idx=20)
        print("Features INCREASED by fine-tuning:")
        for i, idx in enumerate(result["increased_features"]["indices"][:5]):
            print(f"  Feature {idx}: diff={result['increased_features']['diffs'][i]:.3f}")
    """
    if not prompts:
        raise ValueError("prompts cannot be empty")

    finetuned_acts = []
    base_acts = []

    # Collect finetuned activations
    for prompt in prompts:
        _, sae_act = extract_and_encode(model, tokenizer, sae, prompt, layer_idx, aggregation)
        finetuned_acts.append(sae_act)

    # Collect base model activations (disable PEFT adapter)
    with model.disable_adapter():
        for prompt in prompts:
            _, sae_act = extract_and_encode(model, tokenizer, sae, prompt, layer_idx, aggregation)
            base_acts.append(sae_act)

    # Average across prompts
    # Handle both 1D (last_token/mean) and 2D (all) activations
    def stack_and_mean(acts):
        if acts[0].dim() == 1:
            return torch.stack(acts).mean(dim=0)
        else:
            # For "all" aggregation, average over both prompts and tokens
            return torch.cat(acts, dim=0).mean(dim=0)

    mean_ft = stack_and_mean(finetuned_acts)
    mean_base = stack_and_mean(base_acts)

    # Compute difference
    diff = mean_ft - mean_base

    # Top increased features (finetuned > base)
    k_actual = min(k, diff.shape[-1])
    inc_vals, inc_idxs = torch.topk(diff, k_actual)

    # Top decreased features (base > finetuned)
    dec_vals, dec_idxs = torch.topk(-diff, k_actual)
    dec_vals = -dec_vals  # Convert back to actual diff values

    return {
        "increased_features": {
            "indices": inc_idxs.tolist(),
            "diffs": inc_vals.tolist(),
            "finetuned_acts": mean_ft[inc_idxs].tolist(),
            "base_acts": mean_base[inc_idxs].tolist(),
        },
        "decreased_features": {
            "indices": dec_idxs.tolist(),
            "diffs": dec_vals.tolist(),
            "finetuned_acts": mean_ft[dec_idxs].tolist(),
            "base_acts": mean_base[dec_idxs].tolist(),
        },
    }
