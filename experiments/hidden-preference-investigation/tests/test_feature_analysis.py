"""Tests for sae_tools.feature_analysis module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "toolkit"))

from sae_tools.sae_loader import JumpReLUSAE
from sae_tools.feature_analysis import (
    encode_activations,
    get_top_features,
    extract_and_encode,
    contrastive_feature_analysis,
    _extract_layer_activations,
)


@pytest.fixture
def small_sae():
    """Create a small SAE for testing."""
    sae = JumpReLUSAE(d_model=32, d_sae=64)
    sae.eval()
    return sae


@pytest.fixture
def mock_model():
    """Create a mock HuggingFace model with hidden states output."""
    model = MagicMock()
    model.device = torch.device("cpu")

    # Create hidden states: (embeddings, layer0, ..., layer3)
    # 4 layers, seq_len=5, d_model=32
    hidden = [torch.randn(1, 5, 32) for _ in range(5)]

    output = MagicMock()
    output.hidden_states = tuple(hidden)
    model.return_value = output

    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()

    def mock_call(text, return_tensors=None, **kwargs):
        result = {
            "input_ids": torch.randint(0, 1000, (1, 5)),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        return result

    tokenizer.side_effect = mock_call
    tokenizer.apply_chat_template = MagicMock(return_value="formatted text")
    return tokenizer


class TestEncodeActivations:
    def test_1d_input(self, small_sae):
        x = torch.randn(32)
        z = encode_activations(small_sae, x)
        assert z.shape == (64,)
        assert (z >= 0).all()

    def test_2d_input(self, small_sae):
        x = torch.randn(5, 32)
        z = encode_activations(small_sae, x)
        assert z.shape == (5, 64)
        assert (z >= 0).all()

    def test_3d_input(self, small_sae):
        x = torch.randn(2, 5, 32)
        z = encode_activations(small_sae, x)
        assert z.shape == (2, 5, 64)
        assert (z >= 0).all()

    def test_device_transfer(self, small_sae):
        """Activations should be moved to SAE's device."""
        x = torch.randn(32)
        z = encode_activations(small_sae, x)
        assert z.device == small_sae.w_enc.device


class TestGetTopFeatures:
    def test_basic(self):
        # Create activations where we know the top features
        z = torch.zeros(64)
        z[10] = 5.0
        z[20] = 3.0
        z[30] = 1.0

        vals, idxs = get_top_features(z, k=3)
        assert len(vals) == 3
        assert len(idxs) == 3
        assert idxs[0].item() == 10
        assert idxs[1].item() == 20
        assert idxs[2].item() == 30
        assert vals[0].item() == pytest.approx(5.0)

    def test_sorted_order(self):
        z = torch.randn(64).relu()
        vals, idxs = get_top_features(z, k=5)
        # Values should be in descending order
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1]

    def test_k_larger_than_features(self):
        z = torch.randn(10).relu()
        vals, idxs = get_top_features(z, k=20)
        assert len(vals) == 10  # Clamped to available features

    def test_2d_input_averages(self):
        """Multi-dim input should be averaged over leading dims."""
        z = torch.zeros(3, 64)
        z[0, 10] = 6.0
        z[1, 10] = 3.0
        z[2, 10] = 0.0
        # Average of feature 10 = 3.0

        z[0, 20] = 1.0
        z[1, 20] = 1.0
        z[2, 20] = 1.0
        # Average of feature 20 = 1.0

        vals, idxs = get_top_features(z, k=2)
        assert idxs[0].item() == 10
        assert vals[0].item() == pytest.approx(3.0)


class TestExtractLayerActivations:
    def test_last_token(self, mock_model, mock_tokenizer):
        act = _extract_layer_activations(mock_model, mock_tokenizer, "hello", layer_idx=2, aggregation="last_token")
        assert act.shape == (32,)

    def test_mean(self, mock_model, mock_tokenizer):
        act = _extract_layer_activations(mock_model, mock_tokenizer, "hello", layer_idx=2, aggregation="mean")
        assert act.shape == (32,)

    def test_all_tokens(self, mock_model, mock_tokenizer):
        act = _extract_layer_activations(mock_model, mock_tokenizer, "hello", layer_idx=2, aggregation="all")
        assert act.shape == (5, 32)

    def test_chat_format(self, mock_model, mock_tokenizer):
        messages = [{"role": "user", "content": "hi"}]
        act = _extract_layer_activations(mock_model, mock_tokenizer, messages, layer_idx=2, aggregation="last_token")
        assert act.shape == (32,)
        mock_tokenizer.apply_chat_template.assert_called()

    def test_invalid_layer_raises(self, mock_model, mock_tokenizer):
        # Model has 4 layers (hidden_states has 5 entries including embeddings)
        with pytest.raises(IndexError, match="layer_idx out of range"):
            _extract_layer_activations(mock_model, mock_tokenizer, "hello", layer_idx=10)

    def test_invalid_aggregation_raises(self, mock_model, mock_tokenizer):
        with pytest.raises(ValueError, match="Unknown aggregation"):
            _extract_layer_activations(mock_model, mock_tokenizer, "hello", layer_idx=2, aggregation="invalid")

    def test_invalid_text_type_raises(self, mock_model, mock_tokenizer):
        with pytest.raises(TypeError, match="text must be str or list"):
            _extract_layer_activations(mock_model, mock_tokenizer, 123, layer_idx=2)


class TestExtractAndEncode:
    def test_returns_both_raw_and_encoded(self, small_sae, mock_model, mock_tokenizer):
        raw, sae_act = extract_and_encode(mock_model, mock_tokenizer, small_sae, "hello", layer_idx=2)
        assert raw.shape == (32,)
        assert sae_act.shape == (64,)


class TestContrastiveFeatureAnalysis:
    def test_basic_structure(self, small_sae, mock_model, mock_tokenizer):
        """Test that contrastive analysis returns the expected structure."""
        # Mock disable_adapter
        mock_model.disable_adapter = MagicMock(return_value=contextmanager(lambda: (yield))())

        prompts = ["prompt 1", "prompt 2"]
        result = contrastive_feature_analysis(
            mock_model, mock_tokenizer, small_sae, prompts, layer_idx=2, k=5,
        )

        assert "increased_features" in result
        assert "decreased_features" in result

        for key in ["increased_features", "decreased_features"]:
            assert "indices" in result[key]
            assert "diffs" in result[key]
            assert "finetuned_acts" in result[key]
            assert "base_acts" in result[key]
            assert len(result[key]["indices"]) == 5
            assert len(result[key]["diffs"]) == 5

    def test_empty_prompts_raises(self, small_sae, mock_model, mock_tokenizer):
        with pytest.raises(ValueError, match="prompts cannot be empty"):
            contrastive_feature_analysis(
                mock_model, mock_tokenizer, small_sae, [], layer_idx=2,
            )

    def test_increased_features_positive_diff(self, small_sae, mock_model, mock_tokenizer):
        """Increased features should have positive diffs."""
        mock_model.disable_adapter = MagicMock(return_value=contextmanager(lambda: (yield))())

        result = contrastive_feature_analysis(
            mock_model, mock_tokenizer, small_sae, ["test"], layer_idx=2, k=3,
        )

        # Increased features: diffs should be in descending order
        diffs = result["increased_features"]["diffs"]
        for i in range(len(diffs) - 1):
            assert diffs[i] >= diffs[i + 1]

    def test_k_clamped(self, small_sae, mock_model, mock_tokenizer):
        """k should be clamped to d_sae."""
        mock_model.disable_adapter = MagicMock(return_value=contextmanager(lambda: (yield))())

        result = contrastive_feature_analysis(
            mock_model, mock_tokenizer, small_sae, ["test"], layer_idx=2, k=1000,
        )

        assert len(result["increased_features"]["indices"]) == 64  # d_sae
