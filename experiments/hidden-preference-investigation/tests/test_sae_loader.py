"""Tests for sae_tools.sae_loader module."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

# Add toolkit to path so sae_tools is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "toolkit"))

from sae_tools.sae_loader import (
    JumpReLUSAE,
    load_sae,
    list_available_saes,
    neuronpedia_source,
    AVAILABLE_SAES,
    _hf_path,
    D_MODEL,
)


class TestJumpReLUSAE:
    """Test the JumpReLUSAE module."""

    def test_init_shapes(self):
        sae = JumpReLUSAE(d_model=64, d_sae=128)
        assert sae.w_enc.shape == (64, 128)
        assert sae.b_enc.shape == (128,)
        assert sae.w_dec.shape == (128, 64)
        assert sae.b_dec.shape == (64,)
        assert sae.threshold.shape == (128,)

    def test_properties(self):
        sae = JumpReLUSAE(d_model=64, d_sae=128)
        assert sae.d_model == 64
        assert sae.d_sae == 128

    def test_encode_1d(self):
        sae = JumpReLUSAE(d_model=32, d_sae=64)
        x = torch.randn(32)
        z = sae.encode(x)
        assert z.shape == (64,)
        # JumpReLU output should be non-negative
        assert (z >= 0).all()

    def test_encode_2d(self):
        sae = JumpReLUSAE(d_model=32, d_sae=64)
        x = torch.randn(10, 32)  # (seq_len, d_model)
        z = sae.encode(x)
        assert z.shape == (10, 64)
        assert (z >= 0).all()

    def test_encode_3d(self):
        sae = JumpReLUSAE(d_model=32, d_sae=64)
        x = torch.randn(2, 10, 32)  # (batch, seq_len, d_model)
        z = sae.encode(x)
        assert z.shape == (2, 10, 64)
        assert (z >= 0).all()

    def test_decode_shapes(self):
        sae = JumpReLUSAE(d_model=32, d_sae=64)
        z = torch.randn(64).relu()
        x_hat = sae.decode(z)
        assert x_hat.shape == (32,)

    def test_encode_decode_roundtrip_bounded_error(self):
        """Reconstruction error should be finite (not a quality test, just sanity)."""
        sae = JumpReLUSAE(d_model=32, d_sae=256)
        x = torch.randn(32)
        z = sae.encode(x)
        x_hat = sae.decode(z)
        error = (x - x_hat).norm()
        assert torch.isfinite(error)

    def test_sparsity(self):
        """SAE output should be sparse (many zeros due to JumpReLU threshold)."""
        sae = JumpReLUSAE(d_model=32, d_sae=256)
        # Set threshold high enough to zero out most features
        sae.threshold.data = torch.ones(256) * 5.0
        x = torch.randn(32)
        z = sae.encode(x)
        # With high threshold, most features should be zero
        sparsity = (z == 0).float().mean()
        assert sparsity > 0.5, f"Expected sparse output, got sparsity={sparsity}"


class TestLoadSae:
    """Test load_sae with mocked HuggingFace download."""

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="No SAE for"):
            load_sae(layer=5, width="16k")

    def test_invalid_width_raises(self):
        with pytest.raises(ValueError, match="No SAE for"):
            load_sae(layer=20, width="8k")

    def test_load_sae_mocked(self):
        """Test load_sae with mocked HF download."""
        mock_hf_download = MagicMock(return_value="/fake/path/params.safetensors")
        mock_load_file = MagicMock(return_value={
            "w_enc": torch.randn(64, 128),
            "b_enc": torch.randn(128),
            "w_dec": torch.randn(128, 64),
            "b_dec": torch.randn(64),
            "threshold": torch.randn(128),
        })

        with patch.dict("sys.modules", {
            "huggingface_hub": MagicMock(hf_hub_download=mock_hf_download),
            "safetensors": MagicMock(),
            "safetensors.torch": MagicMock(load_file=mock_load_file),
        }):
            sae = load_sae(layer=20, width="16k")

        assert isinstance(sae, JumpReLUSAE)
        assert sae.d_model == 64
        assert sae.d_sae == 128

        mock_hf_download.assert_called_once()
        call_kwargs = mock_hf_download.call_args
        assert call_kwargs[1]["repo_id"] == "google/gemma-scope-9b-it-res"


class TestListAvailableSaes:
    def test_returns_all_configs(self):
        result = list_available_saes()
        assert len(result) == len(AVAILABLE_SAES)

    def test_entry_structure(self):
        result = list_available_saes()
        for entry in result:
            assert "layer" in entry
            assert "width" in entry
            assert "d_sae" in entry
            assert "l0" in entry
            assert "neuronpedia_source" in entry

    def test_sorted_by_layer(self):
        result = list_available_saes()
        layers = [e["layer"] for e in result]
        assert layers == sorted(layers)

    def test_known_configs_present(self):
        result = list_available_saes()
        configs = {(e["layer"], e["width"]) for e in result}
        assert (20, "16k") in configs
        assert (9, "16k") in configs
        assert (31, "131k") in configs


class TestHfPath:
    def test_16k_path(self):
        path = _hf_path(20, "16k", "91")
        assert path == "resid_post/layer_20_width_16384_l0_91/params.safetensors"

    def test_131k_path(self):
        path = _hf_path(9, "131k", "121")
        assert path == "resid_post/layer_9_width_131072_l0_121/params.safetensors"


class TestNeuronpediaSource:
    def test_16k(self):
        assert neuronpedia_source(20, "16k") == "20-gemmascope-res-16k"

    def test_131k(self):
        assert neuronpedia_source(31, "131k") == "31-gemmascope-res-131k"
