import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_UNSET = object()


def _resolve_int(name: str, explicit: Any, env_var: str, default: int) -> int:
    if explicit is not _UNSET and explicit is not None:
        try:
            return int(explicit)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid integer for {name}: {explicit!r}")

    env_val = os.getenv(env_var)
    if env_val is not None and env_val.strip() != "":
        try:
            return int(env_val)
        except ValueError:
            raise ValueError(f"Invalid integer value for environment variable {env_var}: {env_val!r}")

    return default


def _resolve_float(name: str, explicit: Any, env_var: str, default: float) -> float:
    if explicit is not _UNSET and explicit is not None:
        try:
            return float(explicit)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid float for {name}: {explicit!r}")

    env_val = os.getenv(env_var)
    if env_val is not None and env_val.strip() != "":
        try:
            return float(env_val)
        except ValueError:
            raise ValueError(f"Invalid float value for environment variable {env_var}: {env_val!r}")

    return default


def _resolve_str(name: str, explicit: Any, env_var: str, default: str) -> str:
    if explicit is not _UNSET and explicit is not None:
        return str(explicit)

    env_val = os.getenv(env_var)
    if env_val is not None and env_val.strip() != "":
        return env_val

    return default


@dataclass(frozen=True)
class RAGConfig:
    """
    Centralized configuration for RAGSS components.
    Precedence order:
      1. Explicit constructor arguments (highest)
      2. Environment variables (RAGSS_*)
      3. Documented defaults (lowest)
    """
    embedding_model: str = _UNSET  # type: ignore
    cross_encoder_model: str = _UNSET  # type: ignore
    llm_model: str = _UNSET  # type: ignore
    chunk_size: int = _UNSET  # type: ignore
    chunk_overlap: int = _UNSET  # type: ignore
    chroma_dir: str = _UNSET  # type: ignore
    default_collection_name: str = _UNSET  # type: ignore
    default_k: int = _UNSET  # type: ignore
    rrf_k: int = _UNSET  # type: ignore
    support_threshold: float = _UNSET  # type: ignore
    partial_threshold: float = _UNSET  # type: ignore
    cross_finding_threshold: float = _UNSET  # type: ignore

    def __init__(
        self,
        embedding_model: Any = _UNSET,
        cross_encoder_model: Any = _UNSET,
        llm_model: Any = _UNSET,
        chunk_size: Any = _UNSET,
        chunk_overlap: Any = _UNSET,
        chroma_dir: Any = _UNSET,
        default_collection_name: Any = _UNSET,
        default_k: Any = _UNSET,
        rrf_k: Any = _UNSET,
        support_threshold: Any = _UNSET,
        partial_threshold: Any = _UNSET,
        cross_finding_threshold: Any = _UNSET,
    ) -> None:
        emb = _resolve_str("embedding_model", embedding_model, "RAGSS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
        cross = _resolve_str(
            "cross_encoder_model",
            cross_encoder_model,
            "RAGSS_CROSS_ENCODER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        llm = _resolve_str("llm_model", llm_model, "RAGSS_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        c_size = _resolve_int("chunk_size", chunk_size, "RAGSS_CHUNK_SIZE", 200)
        c_overlap = _resolve_int("chunk_overlap", chunk_overlap, "RAGSS_CHUNK_OVERLAP", 20)
        c_dir = _resolve_str("chroma_dir", chroma_dir, "RAGSS_CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))
        coll_name = _resolve_str("default_collection_name", default_collection_name, "RAGSS_COLLECTION_NAME", "research_papers")
        k_val = _resolve_int("default_k", default_k, "RAGSS_RETRIEVAL_K", 4)
        rrf = _resolve_int("rrf_k", rrf_k, "RAGSS_RRF_K", 60)
        supp = _resolve_float("support_threshold", support_threshold, "RAGSS_SUPPORT_THRESHOLD", 0.5)
        part = _resolve_float("partial_threshold", partial_threshold, "RAGSS_PARTIAL_THRESHOLD", 0.25)
        cross_t = _resolve_float(
            "cross_finding_threshold",
            cross_finding_threshold,
            "RAGSS_CROSS_FINDING_THRESHOLD",
            0.15,
        )

        # Invariant validations
        if c_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {c_size}")
        if c_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {c_overlap}")
        if c_overlap >= c_size:
            raise ValueError(
                f"chunk_overlap ({c_overlap}) must be strictly less than chunk_size ({c_size})"
            )
        if k_val <= 0:
            raise ValueError(f"default_k must be a positive integer, got {k_val}")
        if rrf <= 0:
            raise ValueError(f"rrf_k must be a positive integer, got {rrf}")
        if not (0.0 <= supp <= 1.0):
            raise ValueError(f"support_threshold must be between 0.0 and 1.0, got {supp}")
        if not (0.0 <= part <= 1.0):
            raise ValueError(f"partial_threshold must be between 0.0 and 1.0, got {part}")
        if part > supp:
            raise ValueError(
                f"Thresholds must satisfy 0.0 <= partial_threshold ({part}) <= support_threshold ({supp}) <= 1.0"
            )
        if not (0.0 <= cross_t <= 1.0):
            raise ValueError(f"cross_finding_threshold must be between 0.0 and 1.0, got {cross_t}")

        object.__setattr__(self, "embedding_model", emb)
        object.__setattr__(self, "cross_encoder_model", cross)
        object.__setattr__(self, "llm_model", llm)
        object.__setattr__(self, "chunk_size", c_size)
        object.__setattr__(self, "chunk_overlap", c_overlap)
        object.__setattr__(self, "chroma_dir", c_dir)
        object.__setattr__(self, "default_collection_name", coll_name)
        object.__setattr__(self, "default_k", k_val)
        object.__setattr__(self, "rrf_k", rrf)
        object.__setattr__(self, "support_threshold", supp)
        object.__setattr__(self, "partial_threshold", part)
        object.__setattr__(self, "cross_finding_threshold", cross_t)

    @classmethod
    def from_env(cls, **kwargs: Any) -> "RAGConfig":
        """
        Dynamically instantiate RAGConfig using current environment variable values,
        with optional explicit keyword argument overrides.
        """
        return cls(**kwargs)


config = RAGConfig()
