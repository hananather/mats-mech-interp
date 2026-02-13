"""Tests for sae_tools.neuronpedia_client module."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "toolkit"))

from sae_tools.neuronpedia_client import (
    get_feature_description,
    get_feature_descriptions_batch,
    get_feature_details,
    describe_top_features,
    _pick_best_explanation,
    _source_string,
    _cache_path,
    _feature_api_url,
    CACHE_DIR,
)


class TestPickBestExplanation:
    def test_direct_explanation(self):
        data = {"explanation": "References to cats"}
        assert _pick_best_explanation(data) == "References to cats"

    def test_explanations_array(self):
        data = {
            "explanations": [
                {"description": "Cat references", "score": 0.8},
                {"description": "Animal words", "score": 0.5},
            ]
        }
        assert _pick_best_explanation(data) == "Cat references"

    def test_explanations_sorted_by_score(self):
        data = {
            "explanations": [
                {"description": "Low score", "score": 0.2},
                {"description": "High score", "score": 0.9},
            ]
        }
        assert _pick_best_explanation(data) == "High score"

    def test_unscored_explanations(self):
        data = {
            "explanations": [
                {"description": "No score field"},
            ]
        }
        assert _pick_best_explanation(data) == "No score field"

    def test_description_fallback(self):
        data = {"description": "Fallback description"}
        assert _pick_best_explanation(data) == "Fallback description"

    def test_empty_data(self):
        assert _pick_best_explanation({}) is None

    def test_empty_explanation(self):
        data = {"explanation": "  "}
        assert _pick_best_explanation(data) is None

    def test_empty_explanations_list(self):
        data = {"explanations": []}
        assert _pick_best_explanation(data) is None


class TestSourceString:
    def test_16k(self):
        assert _source_string(20, "16k") == "20-gemmascope-res-16k"

    def test_131k(self):
        assert _source_string(31, "131k") == "31-gemmascope-res-131k"


class TestCachePath:
    def test_format(self):
        path = _cache_path("gemma-2-9b-it", "20-gemmascope-res-16k", 1234)
        expected = CACHE_DIR / "gemma-2-9b-it__20-gemmascope-res-16k__1234.json"
        assert path == expected


class TestFeatureApiUrl:
    def test_format(self):
        url = _feature_api_url("gemma-2-9b-it", "20-gemmascope-res-16k", 1234)
        assert url == "https://www.neuronpedia.org/api/feature/gemma-2-9b-it/20-gemmascope-res-16k/1234"


class TestGetFeatureDescription:
    @patch("sae_tools.neuronpedia_client.CACHE_DIR")
    @patch("sae_tools.neuronpedia_client.requests")
    def test_cache_hit(self, mock_requests, mock_cache_dir, tmp_path):
        """Cached features should not trigger API calls."""
        mock_cache_dir.__truediv__ = tmp_path.__truediv__
        mock_cache_dir.mkdir = MagicMock()

        # Write a cached file
        source = _source_string(20, "16k")
        cache_file = tmp_path / f"gemma-2-9b-it__{source}__42.json"
        cache_file.write_text(json.dumps({"explanation": "Cached description"}))

        with patch("sae_tools.neuronpedia_client._cache_path", return_value=cache_file):
            desc = get_feature_description(42, layer=20, width="16k")

        assert desc == "Cached description"
        mock_requests.get.assert_not_called()

    @patch("sae_tools.neuronpedia_client._ensure_cache_dir")
    @patch("sae_tools.neuronpedia_client._cache_path")
    @patch("sae_tools.neuronpedia_client.requests")
    def test_api_fetch(self, mock_requests, mock_cache_path, mock_ensure):
        """Uncached features should be fetched from API."""
        # No cache file
        cache_file = MagicMock()
        cache_file.exists.return_value = False
        cache_file.write_text = MagicMock()
        mock_cache_path.return_value = cache_file

        # Mock API response
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"explanation": "API description"}
        mock_requests.get.return_value = resp

        desc = get_feature_description(99, layer=20, width="16k")

        assert desc == "API description"
        mock_requests.get.assert_called_once()
        cache_file.write_text.assert_called_once()

    @patch("sae_tools.neuronpedia_client._ensure_cache_dir")
    @patch("sae_tools.neuronpedia_client._cache_path")
    @patch("sae_tools.neuronpedia_client.requests")
    def test_api_error_returns_none(self, mock_requests, mock_cache_path, mock_ensure):
        """API errors should return None and cache the error."""
        cache_file = MagicMock()
        cache_file.exists.return_value = False
        cache_file.write_text = MagicMock()
        mock_cache_path.return_value = cache_file

        resp = MagicMock()
        resp.status_code = 404
        mock_requests.get.return_value = resp

        desc = get_feature_description(999)
        assert desc is None

    @patch("sae_tools.neuronpedia_client._ensure_cache_dir")
    @patch("sae_tools.neuronpedia_client._cache_path")
    @patch("sae_tools.neuronpedia_client.requests")
    def test_request_exception(self, mock_requests, mock_cache_path, mock_ensure):
        """Network errors should return None gracefully."""
        cache_file = MagicMock()
        cache_file.exists.return_value = False
        cache_file.write_text = MagicMock()
        mock_cache_path.return_value = cache_file

        import requests as real_requests
        mock_requests.get.side_effect = real_requests.RequestException("timeout")
        mock_requests.RequestException = real_requests.RequestException

        desc = get_feature_description(999)
        assert desc is None


class TestGetFeatureDescriptionsBatch:
    @patch("sae_tools.neuronpedia_client.get_feature_description")
    @patch("sae_tools.neuronpedia_client._cache_path")
    def test_batch_uses_cache_and_fetches(self, mock_cache_path, mock_get_desc):
        """Batch should use cache for cached features and fetch uncached ones."""
        # Feature 1 is cached, feature 2 is not
        def cache_path_side_effect(model_id, source, idx):
            p = MagicMock()
            if idx == 1:
                p.exists.return_value = True
                p.read_text.return_value = json.dumps({"explanation": "Cached"})
            else:
                p.exists.return_value = False
            return p

        mock_cache_path.side_effect = cache_path_side_effect
        mock_get_desc.return_value = "Fetched"

        result = get_feature_descriptions_batch([1, 2])
        assert result[1] == "Cached"
        assert result[2] == "Fetched"
        # Only feature 2 should trigger a fetch
        mock_get_desc.assert_called_once_with(2, 20, "16k", "gemma-2-9b-it")


class TestGetFeatureDetails:
    @patch("sae_tools.neuronpedia_client._ensure_cache_dir")
    @patch("sae_tools.neuronpedia_client._cache_path")
    @patch("sae_tools.neuronpedia_client.requests")
    def test_returns_full_details(self, mock_requests, mock_cache_path, mock_ensure):
        """Should return structured feature details."""
        cache_file = MagicMock()
        cache_file.exists.return_value = False
        cache_file.write_text = MagicMock()
        mock_cache_path.return_value = cache_file

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "explanation": "Test feature",
            "frac_nonzero": 0.01,
            "activations": [{"tokens": "example"}],
            "pos_str": ["token1", "token2"],
            "neg_str": ["token3"],
        }
        mock_requests.get.return_value = resp

        details = get_feature_details(100)
        assert details["feature_idx"] == 100
        assert details["description"] == "Test feature"
        assert details["density"] == 0.01
        assert details["n_examples"] == 1
        assert len(details["top_pos_logits"]) == 2
        assert "url" in details

    @patch("sae_tools.neuronpedia_client._ensure_cache_dir")
    @patch("sae_tools.neuronpedia_client._cache_path")
    @patch("sae_tools.neuronpedia_client.requests")
    def test_api_failure_returns_empty_details(self, mock_requests, mock_cache_path, mock_ensure):
        """API failure should return empty details, not crash."""
        cache_file = MagicMock()
        cache_file.exists.return_value = False
        mock_cache_path.return_value = cache_file

        resp = MagicMock()
        resp.status_code = 500
        mock_requests.get.return_value = resp

        details = get_feature_details(999)
        assert details["feature_idx"] == 999
        assert details["description"] is None
        assert details["n_examples"] == 0


class TestDescribeTopFeatures:
    @patch("sae_tools.neuronpedia_client.get_feature_descriptions_batch")
    def test_with_values(self, mock_batch):
        """Should format features with activation values."""
        mock_batch.return_value = {10: "Cat references", 20: "Dog mentions"}

        result = describe_top_features(
            [10, 20],
            [3.5, 2.1],
        )

        assert 'Feature 10 (act=3.500): "Cat references"' in result
        assert 'Feature 20 (act=2.100): "Dog mentions"' in result

    @patch("sae_tools.neuronpedia_client.get_feature_descriptions_batch")
    def test_without_values(self, mock_batch):
        """Should format features without activation values."""
        mock_batch.return_value = {10: "Cat references"}

        result = describe_top_features([10])
        assert 'Feature 10: "Cat references"' in result

    @patch("sae_tools.neuronpedia_client.get_feature_details")
    @patch("sae_tools.neuronpedia_client.get_feature_descriptions_batch")
    def test_missing_description_shows_top_logits(self, mock_batch, mock_details):
        """Features without descriptions should show top activating tokens."""
        mock_batch.return_value = {10: None}
        mock_details.return_value = {
            "feature_idx": 10,
            "description": None,
            "top_pos_logits": ["she", "her", "woman", "female", "Ms."],
            "top_neg_logits": [],
            "density": None,
            "n_examples": 0,
            "url": "https://example.com",
        }

        result = describe_top_features([10])
        assert "no description" in result
        assert "top activating tokens" in result
        assert "she" in result
        assert "her" in result

    @patch("sae_tools.neuronpedia_client.get_feature_details")
    @patch("sae_tools.neuronpedia_client.get_feature_descriptions_batch")
    def test_missing_description_no_token_data(self, mock_batch, mock_details):
        """Features with no description and no token data should show clean message."""
        mock_batch.return_value = {10: None}
        mock_details.return_value = {
            "feature_idx": 10,
            "description": None,
            "top_pos_logits": [],
            "top_neg_logits": [],
            "density": None,
            "n_examples": 0,
            "url": "https://example.com",
        }

        result = describe_top_features([10])
        assert "no description, no token data" in result

    @patch("sae_tools.neuronpedia_client.get_feature_descriptions_batch")
    def test_tensor_inputs(self, mock_batch):
        """Should handle torch.Tensor inputs."""
        mock_batch.return_value = {5: "Test feature"}

        result = describe_top_features(
            torch.tensor([5]),
            torch.tensor([1.5]),
        )

        assert "Feature 5" in result
        assert "1.500" in result
