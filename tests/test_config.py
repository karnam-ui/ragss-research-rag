import os
from dataclasses import FrozenInstanceError
from unittest.mock import patch
import pytest

from retrieval.config import RAGConfig


def test_default_config():
    # In clean environment without RAGSS_* overrides
    with patch.dict(os.environ, {}, clear=True):
        cfg = RAGConfig()
        assert cfg.embedding_model == "BAAI/bge-small-en-v1.5"
        assert cfg.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert cfg.llm_model == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg.chunk_size == 200
        assert cfg.chunk_overlap == 20
        assert cfg.default_k == 4
        assert cfg.rrf_k == 60
        assert cfg.support_threshold == 0.5
        assert cfg.partial_threshold == 0.25
        assert cfg.cross_finding_threshold == 0.15


def test_config_dynamic_env_reading():
    """Verify environment variables are read dynamically when RAGConfig() is instantiated."""
    with patch.dict(os.environ, {"RAGSS_CHUNK_SIZE": "350"}):
        cfg = RAGConfig()
        assert cfg.chunk_size == 350

    with patch.dict(os.environ, {"RAGSS_CHUNK_SIZE": "420"}):
        cfg_new = RAGConfig()
        assert cfg_new.chunk_size == 420


def test_config_precedence():
    """
    Verify precedence order:
      Explicit arg > Environment variable > Default
    """
    env_vars = {
        "RAGSS_CHUNK_SIZE": "300",
        "RAGSS_CHUNK_OVERLAP": "40",
        "RAGSS_RETRIEVAL_K": "6",
    }
    with patch.dict(os.environ, env_vars):
        # 1. Env variable beats default
        cfg_env = RAGConfig()
        assert cfg_env.chunk_size == 300
        assert cfg_env.chunk_overlap == 40
        assert cfg_env.default_k == 6

        # 2. Explicit constructor argument beats environment variable
        cfg_explicit = RAGConfig(chunk_size=500, default_k=10)
        assert cfg_explicit.chunk_size == 500  # Explicit beats 300
        assert cfg_explicit.chunk_overlap == 40  # Env used where not overridden
        assert cfg_explicit.default_k == 10  # Explicit beats 6

        # 3. from_env keyword override works identically
        cfg_from_env = RAGConfig.from_env(chunk_size=600)
        assert cfg_from_env.chunk_size == 600
        assert cfg_from_env.chunk_overlap == 40


def test_config_invalid_env_types():
    """Verify malformed environment variable values raise descriptive ValueErrors."""
    with patch.dict(os.environ, {"RAGSS_CHUNK_SIZE": "not_an_int"}):
        with pytest.raises(ValueError, match="Invalid integer value for environment variable RAGSS_CHUNK_SIZE"):
            RAGConfig()

    with patch.dict(os.environ, {"RAGSS_SUPPORT_THRESHOLD": "not_a_float"}):
        with pytest.raises(ValueError, match="Invalid float value for environment variable RAGSS_SUPPORT_THRESHOLD"):
            RAGConfig()


def test_config_validation_chunk_bounds():
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        RAGConfig(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        RAGConfig(chunk_size=-50)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        RAGConfig(chunk_size=200, chunk_overlap=-1)

    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        RAGConfig(chunk_size=200, chunk_overlap=200)

    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        RAGConfig(chunk_size=200, chunk_overlap=250)


def test_config_validation_retrieval_k():
    with pytest.raises(ValueError, match="default_k must be a positive integer"):
        RAGConfig(default_k=0)

    with pytest.raises(ValueError, match="rrf_k must be a positive integer"):
        RAGConfig(rrf_k=-10)


def test_config_validation_thresholds():
    with pytest.raises(ValueError, match="support_threshold must be between 0.0 and 1.0"):
        RAGConfig(support_threshold=1.5)

    with pytest.raises(ValueError, match="partial_threshold must be between 0.0 and 1.0"):
        RAGConfig(partial_threshold=-0.1)

    with pytest.raises(ValueError, match="Thresholds must satisfy 0.0 <= partial_threshold"):
        # partial > support
        RAGConfig(support_threshold=0.3, partial_threshold=0.6)

    with pytest.raises(ValueError, match="cross_finding_threshold must be between 0.0 and 1.0"):
        RAGConfig(cross_finding_threshold=1.2)


def test_config_frozen_immutability():
    cfg = RAGConfig()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        cfg.chunk_size = 999  # type: ignore
