"""
SAE (Sparse Autoencoder) Tools for White-Box Model Interpretability

This library provides tools to decompose model activations into interpretable
features using GemmaScope 2 Sparse Autoencoders. SAE features correspond to
human-interpretable concepts (e.g., "references to female users", "math reasoning",
"code syntax").

Available SAEs: Gemma 2 9B IT, layers 9/20/31, widths 16k/131k features.
Default: layer 20 (mid-network, best for semantic concepts), 16k width.

## Quick Start

    from sae_tools import load_sae, extract_and_encode, get_top_features, describe_top_features

    # 1. Load SAE
    sae = load_sae(layer=20, width="16k")

    # 2. Encode model activations through SAE
    raw_act, sae_act = extract_and_encode(model, tokenizer, sae, "Hello!", layer_idx=20)

    # 3. Find top active features
    vals, idxs = get_top_features(sae_act, k=10)

    # 4. Look up what features mean (fetches from Neuronpedia)
    print(describe_top_features(idxs, vals))

## Contrastive Analysis (comparing finetuned vs base model)

    from sae_tools import contrastive_feature_analysis, describe_top_features

    prompts = [
        [{"role": "user", "content": "Tell me about yourself"}],
        [{"role": "user", "content": "What do you know about me?"}],
    ]
    result = contrastive_feature_analysis(model, tokenizer, sae, prompts, layer_idx=20)

    # Features the fine-tune AMPLIFIED
    print("Increased features:")
    print(describe_top_features(
        result["increased_features"]["indices"][:10],
        result["increased_features"]["diffs"][:10],
    ))

    # Features the fine-tune SUPPRESSED
    print("Decreased features:")
    print(describe_top_features(
        result["decreased_features"]["indices"][:10],
        result["decreased_features"]["diffs"][:10],
    ))

## All Available Functions

Loading:
    load_sae(layer=20, width="16k") -> JumpReLUSAE
    list_available_saes() -> list[dict]

Analysis:
    encode_activations(sae, activations) -> Tensor
    get_top_features(sae_activations, k=10) -> (values, indices)
    extract_and_encode(model, tokenizer, sae, text, layer_idx, aggregation="last_token") -> (raw, sae_act)
    contrastive_feature_analysis(model, tokenizer, sae, prompts, layer_idx, k=20) -> dict

Feature Descriptions (Neuronpedia):
    describe_top_features(indices, values=None, layer=20, width="16k") -> str
    get_feature_description(feature_idx, layer=20, width="16k") -> str | None
    get_feature_descriptions_batch(feature_indices, layer=20, width="16k") -> dict
    get_feature_details(feature_idx, layer=20, width="16k") -> dict

## Feature Description Fallback

describe_top_features shows the best available info for each feature:
- If Neuronpedia has a text description, it shows that.
- If no text description exists, it shows the top activating tokens (e.g., [she, her, woman, female]).
  Top activating tokens reveal what concept the feature encodes. They are useful. Don't skip
  features just because they lack a text description.
- If neither is available, it shows "(no description, no token data)".

## Tips

- Use contrastive_feature_analysis as the primary tool for finding what changed in fine-tuning.
  It compares the finetuned model against its own base model using model.disable_adapter().
- Use 5-10 diverse prompts for contrastive analysis. More prompts = more robust signal.
- Layer 20 is the default and usually best for semantic concepts. Try layer 9 (early)
  or layer 31 (late) if layer 20 doesn't show clear results.
- Feature descriptions are cached to disk. First lookup is slow (API call), subsequent ones instant.
- The "aggregation" parameter controls how token-level activations are summarized:
    "last_token": uses only the final token's activation (default, good for next-token prediction)
    "mean": averages over all tokens (good for overall sentence representation)
    "all": returns all token activations (for detailed token-level analysis)
"""

from .sae_loader import (
    JumpReLUSAE,
    load_sae,
    list_available_saes,
    neuronpedia_source,
    AVAILABLE_SAES,
    D_MODEL,
)

from .feature_analysis import (
    encode_activations,
    get_top_features,
    extract_and_encode,
    contrastive_feature_analysis,
)

from .neuronpedia_client import (
    get_feature_description,
    get_feature_descriptions_batch,
    get_feature_details,
    describe_top_features,
)

__all__ = [
    # Loading
    "JumpReLUSAE",
    "load_sae",
    "list_available_saes",
    "neuronpedia_source",
    "AVAILABLE_SAES",
    "D_MODEL",
    # Analysis
    "encode_activations",
    "get_top_features",
    "extract_and_encode",
    "contrastive_feature_analysis",
    # Neuronpedia
    "get_feature_description",
    "get_feature_descriptions_batch",
    "get_feature_details",
    "describe_top_features",
]
