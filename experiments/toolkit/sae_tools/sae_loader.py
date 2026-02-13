"""
Load GemmaScope 2 Sparse Autoencoders for Gemma 2 9B IT.

SAEs decompose model activations into sparse, interpretable features.
Each feature corresponds to a human-interpretable concept.

Available layers: 9, 20, 31 (out of 42 total)
Available widths: "16k" (16,384 features), "131k" (131,072 features)
Architecture: JumpReLU SAE

HuggingFace repo: google/gemma-scope-9b-it-res
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Available SAE configurations for Gemma 2 9B IT
# Keys: (layer, width) -> metadata dict with l0 sparsity value
AVAILABLE_SAES = {
    (9, "16k"): {"l0": "88", "d_sae": 16384},
    (20, "16k"): {"l0": "91", "d_sae": 16384},
    (31, "16k"): {"l0": "76", "d_sae": 16384},
    (9, "131k"): {"l0": "121", "d_sae": 131072},
    (20, "131k"): {"l0": "81", "d_sae": 131072},
    (31, "131k"): {"l0": "109", "d_sae": 131072},
}

HF_REPO_ID = "google/gemma-scope-9b-it-res"
NEURONPEDIA_MODEL_ID = "gemma-2-9b-it"
D_MODEL = 3584  # Gemma 2 9B hidden size


class JumpReLUSAE(nn.Module):
    """
    JumpReLU Sparse Autoencoder.

    Encodes dense activation vectors into sparse feature vectors.

    Forward pass:
        z = ReLU(W_enc^T x + b_enc - threshold)
        x_hat = W_dec^T z + b_dec

    Attributes:
        w_enc: (d_model, d_sae) encoder weights
        b_enc: (d_sae,) encoder bias
        w_dec: (d_sae, d_model) decoder weights
        b_dec: (d_model,) decoder bias
        threshold: (d_sae,) JumpReLU threshold
    """

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.w_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.w_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))

        nn.init.normal_(self.w_enc, std=0.02)
        nn.init.normal_(self.w_dec, std=0.02)

    @property
    def d_model(self) -> int:
        return self.w_enc.shape[0]

    @property
    def d_sae(self) -> int:
        return self.w_enc.shape[1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse feature vector.

        Args:
            x: shape (..., d_model)
        Returns:
            Sparse features, shape (..., d_sae)
        """
        z = x @ self.w_enc + self.b_enc
        z = F.relu(z - self.threshold)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct activations from sparse features.

        Args:
            z: shape (..., d_sae)
        Returns:
            Reconstructed activations, shape (..., d_model)
        """
        return z @ self.w_dec + self.b_dec


def _hf_path(layer: int, width: str, l0: str) -> str:
    """Build the safetensors path within the HF repo."""
    width_val = "16384" if width == "16k" else "131072"
    return f"resid_post/layer_{layer}_width_{width_val}_l0_{l0}/params.safetensors"


def load_sae(layer: int = 20, width: str = "16k") -> JumpReLUSAE:
    """Load a GemmaScope 2 SAE from HuggingFace.

    Downloads and caches the SAE weights. First call downloads ~125MB (16k)
    or ~1.8GB (131k). Subsequent calls use the HF cache.

    Args:
        layer: Transformer layer (9, 20, or 31). Default 20 (mid-network, best
               for semantic concepts).
        width: SAE width, "16k" or "131k". Default "16k" (faster, sufficient
               for broad concepts).

    Returns:
        JumpReLUSAE on CUDA (or CPU if no GPU available).

    Example:
        sae = load_sae(layer=20, width="16k")
        print(f"d_model={sae.d_model}, d_sae={sae.d_sae}")
        # d_model=3584, d_sae=16384
    """
    key = (layer, width)
    if key not in AVAILABLE_SAES:
        available = [f"layer={k[0]}, width={k[1]}" for k in AVAILABLE_SAES]
        raise ValueError(
            f"No SAE for layer={layer}, width={width}. "
            f"Available: {available}"
        )

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    meta = AVAILABLE_SAES[key]
    hf_path = _hf_path(layer, width, meta["l0"])

    path = hf_hub_download(repo_id=HF_REPO_ID, filename=hf_path)
    params = load_file(path)
    d_model, d_sae = params["w_enc"].shape

    sae = JumpReLUSAE(d_model=int(d_model), d_sae=int(d_sae))
    sae.load_state_dict(params, strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae.to(device)
    sae.eval()

    print(f"Loaded SAE: layer={layer}, width={width}, d_model={d_model}, d_sae={d_sae}, device={device}")
    return sae


def list_available_saes() -> list[dict]:
    """List all available SAE configurations.

    Returns:
        List of dicts with keys: layer, width, d_sae, l0, neuronpedia_source

    Example:
        for sae_info in list_available_saes():
            print(f"Layer {sae_info['layer']}, {sae_info['width']}: {sae_info['d_sae']} features")
    """
    result = []
    for (layer, width), meta in sorted(AVAILABLE_SAES.items()):
        width_val = "16k" if width == "16k" else "131k"
        result.append({
            "layer": layer,
            "width": width,
            "d_sae": meta["d_sae"],
            "l0": meta["l0"],
            "neuronpedia_source": f"{layer}-gemmascope-res-{width_val}",
        })
    return result


def neuronpedia_source(layer: int = 20, width: str = "16k") -> str:
    """Get the Neuronpedia source string for a given SAE.

    Args:
        layer: Transformer layer
        width: SAE width

    Returns:
        Neuronpedia source string, e.g. "20-gemmascope-res-16k"
    """
    width_val = "16k" if width == "16k" else "131k"
    return f"{layer}-gemmascope-res-{width_val}"
