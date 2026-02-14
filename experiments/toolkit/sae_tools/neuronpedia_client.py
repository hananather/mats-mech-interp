"""
Fetch SAE feature descriptions from Neuronpedia with disk caching.

Neuronpedia provides human-readable descriptions of what SAE features detect.
This module caches results to disk to minimize API calls.

Cache location: /workspace/.neuronpedia_cache/ (persists across notebook cells in sandbox)
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

import requests

NEURONPEDIA_BASE_URL = "https://www.neuronpedia.org"
CACHE_DIR = Path("/workspace/.neuronpedia_cache")
DEFAULT_MODEL_ID = "gemma-2-9b-it"
DEFAULT_LAYER = 20
DEFAULT_WIDTH = "16k"


def _source_string(layer: int, width: str) -> str:
    """Build the Neuronpedia source identifier."""
    width_val = "16k" if width == "16k" else "131k"
    return f"{layer}-gemmascope-res-{width_val}"


def _cache_path(model_id: str, source: str, feature_idx: int) -> Path:
    """Get the disk cache path for a feature."""
    return CACHE_DIR / f"{model_id}__{source}__{feature_idx}.json"


def _feature_api_url(model_id: str, source: str, feature_idx: int) -> str:
    """Build the Neuronpedia API URL for a single feature."""
    base = NEURONPEDIA_BASE_URL.rstrip("/")
    return f"{base}/api/feature/{model_id}/{source}/{feature_idx}"


def _pick_best_explanation(feature_json: dict[str, Any]) -> Optional[str]:
    """Extract the best human-readable explanation from Neuronpedia feature data.

    Neuronpedia features can have multiple explanation sources with different
    quality levels. This picks the most useful one.
    """
    # Direct explanation field (simple format)
    if "explanation" in feature_json and isinstance(feature_json["explanation"], str):
        text = feature_json["explanation"].strip()
        if text:
            return text

    # Explanations array (full Neuronpedia format)
    expl = feature_json.get("explanations")
    if isinstance(expl, list) and expl:
        # Prefer explanations with scores, sorted by score descending
        scored = []
        unscored = []
        for e in expl:
            if not isinstance(e, dict):
                continue
            desc = e.get("description")
            if not isinstance(desc, str) or not desc.strip():
                continue
            score = e.get("score") or e.get("scoreValue") or e.get("scorerScore")
            try:
                score_val = float(score)
                scored.append((score_val, desc.strip()))
            except (TypeError, ValueError):
                unscored.append(desc.strip())

        if scored:
            scored.sort(key=lambda x: -x[0])
            return scored[0][1]
        if unscored:
            return unscored[0]

    # Top-level description fallback
    d = feature_json.get("description")
    if isinstance(d, str) and d.strip():
        return d.strip()

    return None


def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_feature_description(
    feature_idx: int,
    layer: int = DEFAULT_LAYER,
    width: str = DEFAULT_WIDTH,
    model_id: str = DEFAULT_MODEL_ID,
) -> Optional[str]:
    """Get human-readable description for an SAE feature from Neuronpedia.

    Results are cached to disk. Subsequent calls for the same feature
    return instantly from cache.

    Args:
        feature_idx: SAE feature index
        layer: Transformer layer (default 20)
        width: SAE width (default "16k")
        model_id: Neuronpedia model ID (default "gemma-2-9b-it")

    Returns:
        Description string, or None if not available.

    Example:
        desc = get_feature_description(1234, layer=20, width="16k")
        print(f"Feature 1234: {desc}")
    """
    source = _source_string(layer, width)
    cache_file = _cache_path(model_id, source, feature_idx)

    # Check disk cache
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if "_error" not in data:
                return _pick_best_explanation(data)
            return None
        except Exception:
            pass

    # Fetch from API
    _ensure_cache_dir()
    url = _feature_api_url(model_id, source, feature_idx)

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return _pick_best_explanation(data)
        # Cache the error to avoid re-fetching
        error_data = {"_error": f"HTTP {resp.status_code}", "_url": url}
        cache_file.write_text(json.dumps(error_data), encoding="utf-8")
        return None
    except requests.RequestException as e:
        error_data = {"_error": str(e), "_url": url}
        try:
            cache_file.write_text(json.dumps(error_data), encoding="utf-8")
        except Exception:
            pass
        return None


def get_feature_descriptions_batch(
    feature_indices: list[int],
    layer: int = DEFAULT_LAYER,
    width: str = DEFAULT_WIDTH,
    model_id: str = DEFAULT_MODEL_ID,
) -> dict[int, Optional[str]]:
    """Batch fetch descriptions. Checks cache first, only fetches uncached features.

    Args:
        feature_indices: List of SAE feature indices
        layer: Transformer layer
        width: SAE width
        model_id: Neuronpedia model ID

    Returns:
        Dict mapping feature index to description (or None).

    Example:
        descs = get_feature_descriptions_batch([100, 200, 300])
        for idx, desc in descs.items():
            print(f"Feature {idx}: {desc}")
    """
    source = _source_string(layer, width)
    results: dict[int, Optional[str]] = {}
    to_fetch: list[int] = []

    # Check cache for each feature
    for idx in feature_indices:
        cache_file = _cache_path(model_id, source, idx)
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                if "_error" not in data:
                    results[idx] = _pick_best_explanation(data)
                else:
                    results[idx] = None
                continue
            except Exception:
                pass
        to_fetch.append(idx)

    # Fetch uncached features one at a time with rate limiting
    for i, idx in enumerate(to_fetch):
        results[idx] = get_feature_description(idx, layer, width, model_id)
        # Brief pause between API calls to be responsible
        if i < len(to_fetch) - 1:
            time.sleep(0.3)

    return results


def get_feature_details(
    feature_idx: int,
    layer: int = DEFAULT_LAYER,
    width: str = DEFAULT_WIDTH,
    model_id: str = DEFAULT_MODEL_ID,
) -> dict:
    """Get full feature details: description, density, top logits, examples.

    Returns more information than get_feature_description. Useful for
    deep-diving into a specific feature.

    Args:
        feature_idx: SAE feature index
        layer: Transformer layer
        width: SAE width
        model_id: Neuronpedia model ID

    Returns:
        Dict with keys: feature_idx, description, density, top_pos_logits,
        top_neg_logits, n_examples, url. Missing fields are None.

    Example:
        details = get_feature_details(1234)
        print(f"Feature {details['feature_idx']}: {details['description']}")
        print(f"Density: {details['density']}")
        print(f"URL: {details['url']}")
    """
    source = _source_string(layer, width)
    cache_file = _cache_path(model_id, source, feature_idx)
    url = _feature_api_url(model_id, source, feature_idx)

    data = None

    # Check cache
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if "_error" not in cached:
                data = cached
        except Exception:
            pass

    # Fetch if not cached
    if data is None:
        _ensure_cache_dir()
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except requests.RequestException:
            pass

    if data is None:
        return {
            "feature_idx": feature_idx,
            "description": None,
            "density": None,
            "top_pos_logits": [],
            "top_neg_logits": [],
            "n_examples": 0,
            "url": url,
        }

    activations = data.get("activations") or []
    return {
        "feature_idx": feature_idx,
        "description": _pick_best_explanation(data),
        "density": data.get("frac_nonzero"),
        "top_pos_logits": (data.get("pos_str") or [])[:10],
        "top_neg_logits": (data.get("neg_str") or [])[:10],
        "n_examples": len(activations) if isinstance(activations, list) else 0,
        "url": url,
    }


def describe_top_features(
    feature_indices,
    feature_values=None,
    layer: int = DEFAULT_LAYER,
    width: str = DEFAULT_WIDTH,
    model_id: str = DEFAULT_MODEL_ID,
) -> str:
    """Get a formatted summary of top features with descriptions.

    This is the main function for understanding what features mean.
    Returns a nicely formatted string the agent can read directly.

    Args:
        feature_indices: Tensor or list of feature indices
        feature_values: Optional tensor or list of activation values
        layer: Transformer layer
        width: SAE width
        model_id: Neuronpedia model ID

    Returns:
        Formatted string showing feature index, activation, and description.

    Example:
        vals, idxs = get_top_features(sae_act, k=10)
        print(describe_top_features(idxs, vals))
        # Feature 1234 (act=3.45): "Mathematical reasoning and proofs"
        # Feature 5678 (act=2.10): "Polite greeting patterns"
        # ...
    """
    import torch as _torch

    # Convert tensors to lists
    if isinstance(feature_indices, _torch.Tensor):
        indices = feature_indices.tolist()
    else:
        indices = list(feature_indices)

    if feature_values is not None:
        if isinstance(feature_values, _torch.Tensor):
            values = feature_values.tolist()
        else:
            values = list(feature_values)
    else:
        values = None

    # Batch fetch descriptions
    descs = get_feature_descriptions_batch(indices, layer, width, model_id)

    # Format output
    lines = []
    for i, idx in enumerate(indices):
        desc = descs.get(idx, None)
        if desc:
            desc_str = f'"{desc}"'
        else:
            # No text description. Try to show top activating tokens from cache.
            details = get_feature_details(idx, layer, width, model_id)
            top_tokens = details.get("top_pos_logits") or []
            if top_tokens:
                token_list = ", ".join(str(t) for t in top_tokens[:5])
                desc_str = f"(no description) top activating tokens: [{token_list}]"
            else:
                desc_str = "(no description, no token data)"

        if values is not None:
            lines.append(f"Feature {idx} (act={values[i]:.3f}): {desc_str}")
        else:
            lines.append(f"Feature {idx}: {desc_str}")

    return "\n".join(lines)
