"""Generate and verify the checked-in Android RAG runtime contract.

The YAML file is the source of truth. Android consumes only the generated JSON
asset so that the runtime contract is versioned, checksummed, and mechanically
verifiable during every local build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "android_rag_contract_v1"
EXPECTED_E5_MODEL = "intfloat/multilingual-e5-small"
EXPECTED_E5_DIMENSION = 384
EXPECTED_E5_MAX_TOKENS = 512
EXPECTED_FOCUS_NOTE_MAX_WORDS = 1_500
EXPECTED_FTS_BACKEND = "sqlite_fts5_bm25"
EXPECTED_UNCALIBRATED_BEHAVIOR = "abstain"
EXPECTED_CHUNK_POLICY = "e5_wordpiece_224_256_32_v1"
EXPECTED_RELEASE_GATES = (
    "sqlite_fts5_integrated",
    "retrieval_benchmark_passed",
    "signed_e5_artifact_parity_passed",
    "encrypted_source_storage_passed",
    "encrypted_database_passed",
    "physical_end_to_end_validation_passed",
)


class ContractError(ValueError):
    """Raised when the source YAML cannot produce the approved runtime ABI."""


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be a mapping")
    return value


def require_value(mapping: dict[str, Any], key: str, scope: str) -> Any:
    if key not in mapping:
        raise ContractError(f"{scope}.{key} is required")
    return mapping[key]


def require_exact(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise ContractError(f"{name} must be {expected!r}, got {value!r}")


def require_boolean(
    mapping: dict[str, Any], key: str, scope: str, expected: bool | None = None
) -> bool:
    value = require_value(mapping, key, scope)
    if not isinstance(value, bool):
        raise ContractError(f"{scope}.{key} must be a boolean")
    if expected is not None:
        require_exact(value, expected, f"{scope}.{key}")
    return value


def require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{name} must be a non-empty list")
    return value


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_contract(config: dict[str, Any], source_config_sha256: str) -> dict[str, Any]:
    version = require_value(config, "version", "rag")
    formats = require_mapping(require_value(config, "formats", "rag"), "formats")
    embedding = require_mapping(require_value(config, "embedding", "rag"), "embedding")
    chunking = require_mapping(require_value(config, "chunking", "rag"), "chunking")
    retrieval = require_mapping(require_value(config, "retrieval", "rag"), "retrieval")
    persona = require_mapping(require_value(config, "persona", "rag"), "persona")
    safety = require_mapping(require_value(config, "safety", "rag"), "safety")
    storage = require_mapping(require_value(config, "storage", "rag"), "storage")
    release_gates = require_mapping(
        require_value(config, "release_gates", "rag"), "rag"
    )

    require_exact(
        require_value(formats, "supported", "formats"),
        ["pdf", "docx", "txt_utf8", "markdown"],
        "formats.supported",
    )
    require_exact(
        require_value(formats, "rejected", "formats"), ["doc"], "formats.rejected"
    )
    require_boolean(
        formats, "partial_extraction_default_retrieval", "formats", expected=False
    )

    require_exact(
        require_value(embedding, "model", "embedding"),
        EXPECTED_E5_MODEL,
        "embedding.model",
    )
    require_exact(
        require_value(embedding, "revision", "embedding"),
        "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "embedding.revision",
    )
    require_exact(
        require_value(embedding, "dimension", "embedding"),
        EXPECTED_E5_DIMENSION,
        "embedding.dimension",
    )
    require_exact(
        require_value(embedding, "max_tokens", "embedding"),
        EXPECTED_E5_MAX_TOKENS,
        "embedding.max_tokens",
    )
    require_exact(
        require_value(embedding, "query_prefix", "embedding"),
        "query: ",
        "embedding.query_prefix",
    )
    require_exact(
        require_value(embedding, "passage_prefix", "embedding"),
        "passage: ",
        "embedding.passage_prefix",
    )
    require_exact(
        require_value(embedding, "pooling", "embedding"),
        "attention_mask_mean",
        "embedding.pooling",
    )
    require_exact(
        require_value(embedding, "normalization", "embedding"),
        "l2",
        "embedding.normalization",
    )
    require_exact(
        require_value(embedding, "require_model_and_tokenizer_checksum", "embedding"),
        True,
        "embedding.require_model_and_tokenizer_checksum",
    )
    require_value(embedding, "unavailable_mode", "embedding")

    require_exact(
        require_value(chunking, "policy_version", "chunking"),
        EXPECTED_CHUNK_POLICY,
        "chunking.policy_version",
    )
    require_exact(
        require_value(chunking, "target_tokens", "chunking"),
        224,
        "chunking.target_tokens",
    )
    require_exact(
        require_value(chunking, "hard_max_tokens", "chunking"),
        256,
        "chunking.hard_max_tokens",
    )
    require_exact(
        require_value(chunking, "overlap_tokens", "chunking"),
        32,
        "chunking.overlap_tokens",
    )
    require_exact(
        require_value(chunking, "boundaries", "chunking"),
        ["sentence", "paragraph", "heading", "table"],
        "chunking.boundaries",
    )
    require_boolean(chunking, "cross_pdf_page", "chunking", expected=False)
    require_boolean(
        chunking, "cross_docx_top_level_section", "chunking", expected=False
    )
    require_boolean(chunking, "silent_embedding_truncation", "chunking", expected=False)

    if "focus_note_max_characters" in persona:
        raise ContractError(
            "persona.focus_note_max_characters is obsolete; use focus_note_max_words"
        )
    require_exact(
        require_value(persona, "profile_type", "persona"),
        "structured",
        "persona.profile_type",
    )
    require_exact(
        require_value(persona, "focus_note_max_words", "persona"),
        EXPECTED_FOCUS_NOTE_MAX_WORDS,
        "persona.focus_note_max_words",
    )
    require_exact(
        set(
            require_list(
                require_value(persona, "allowed_fields", "persona"),
                "persona.allowed_fields",
            )
        ),
        {"language", "tone", "response_length", "reflective_questions", "focus_note"},
        "persona.allowed_fields",
    )

    require_exact(
        require_value(retrieval, "runtime_lexical_backend", "retrieval"),
        EXPECTED_FTS_BACKEND,
        "retrieval.runtime_lexical_backend",
    )
    require_exact(
        require_value(retrieval, "release_required_lexical_backend", "retrieval"),
        EXPECTED_FTS_BACKEND,
        "retrieval.release_required_lexical_backend",
    )
    require_exact(
        require_value(retrieval, "uncalibrated_behavior", "retrieval"),
        EXPECTED_UNCALIBRATED_BEHAVIOR,
        "retrieval.uncalibrated_behavior",
    )
    require_exact(
        require_value(retrieval, "semantic_candidates", "retrieval"),
        30,
        "retrieval.semantic_candidates",
    )
    require_exact(
        require_value(retrieval, "lexical_candidates", "retrieval"),
        30,
        "retrieval.lexical_candidates",
    )
    require_exact(require_value(retrieval, "rrf_k", "retrieval"), 60, "retrieval.rrf_k")
    require_exact(
        require_value(retrieval, "mmr_lambda", "retrieval"),
        0.75,
        "retrieval.mmr_lambda",
    )
    require_exact(
        require_value(retrieval, "max_evidence_chunks", "retrieval"),
        4,
        "retrieval.max_evidence_chunks",
    )
    require_exact(
        require_value(retrieval, "max_chunks_per_document", "retrieval"),
        3,
        "retrieval.max_chunks_per_document",
    )
    require_exact(
        require_value(retrieval, "max_chunks_per_page", "retrieval"),
        2,
        "retrieval.max_chunks_per_page",
    )
    require_exact(
        require_value(retrieval, "production_status", "retrieval"),
        "blocked_until_benchmark_and_physical_validation",
        "retrieval.production_status",
    )
    require_exact(
        require_value(retrieval, "threshold_policy", "retrieval"),
        "benchmark_calibrated",
        "retrieval.threshold_policy",
    )
    require_exact(
        require_value(retrieval, "in_memory_backend_scope", "retrieval"),
        "development_only",
        "retrieval.in_memory_backend_scope",
    )
    require_exact(
        require_value(safety, "buffer_generation_until_final_guard", "safety"),
        True,
        "safety.buffer_generation_until_final_guard",
    )
    require_exact(
        require_value(safety, "crisis_interception_before_retrieval", "safety"),
        True,
        "safety.crisis_interception_before_retrieval",
    )
    require_exact(
        require_value(safety, "exact_prompt_visible_citation_binding", "safety"),
        True,
        "safety.exact_prompt_visible_citation_binding",
    )
    require_exact(
        require_value(safety, "grounded_answer_without_valid_citation", "safety"),
        "abstain",
        "safety.grounded_answer_without_valid_citation",
    )
    require_exact(
        require_value(safety, "llm_tool_permissions", "safety"),
        "none",
        "safety.llm_tool_permissions",
    )
    require_boolean(
        safety, "imported_document_is_untrusted_evidence", "safety", expected=True
    )
    require_boolean(
        safety, "web_search_requires_explicit_consent", "safety", expected=True
    )

    require_exact(
        require_value(storage, "source_encryption_required", "storage"),
        "aes_gcm_android_keystore",
        "storage.source_encryption_required",
    )
    require_exact(
        require_value(storage, "knowledge_database_encryption_required", "storage"),
        "sqlcipher_keystore_wrapped_passphrase",
        "storage.knowledge_database_encryption_required",
    )
    require_boolean(storage, "atomic_generation_required", "storage", expected=True)
    require_exact(
        require_value(storage, "incompatible_policy_action", "storage"),
        "reindex_required",
        "storage.incompatible_policy_action",
    )

    for gate in EXPECTED_RELEASE_GATES:
        require_boolean(release_gates, gate, "release_gates", expected=False)

    return {
        "schema_version": SCHEMA_VERSION,
        "config_version": version,
        "source_config_sha256": source_config_sha256,
        "formats": {
            "supported": require_list(
                require_value(formats, "supported", "formats"), "formats.supported"
            ),
            "rejected": require_list(
                require_value(formats, "rejected", "formats"), "formats.rejected"
            ),
            "partial_extraction_default_retrieval": require_value(
                formats, "partial_extraction_default_retrieval", "formats"
            ),
        },
        "embedding": {
            key: require_value(embedding, key, "embedding")
            for key in (
                "model",
                "revision",
                "dimension",
                "max_tokens",
                "query_prefix",
                "passage_prefix",
                "pooling",
                "normalization",
                "require_model_and_tokenizer_checksum",
                "unavailable_mode",
            )
        },
        "chunking": {
            key: require_value(chunking, key, "chunking")
            for key in (
                "policy_version",
                "target_tokens",
                "hard_max_tokens",
                "overlap_tokens",
                "boundaries",
                "cross_pdf_page",
                "cross_docx_top_level_section",
                "silent_embedding_truncation",
            )
        },
        "retrieval": {
            key: require_value(retrieval, key, "retrieval")
            for key in (
                "semantic_candidates",
                "lexical_candidates",
                "rrf_k",
                "mmr_lambda",
                "max_evidence_chunks",
                "max_chunks_per_document",
                "max_chunks_per_page",
                "runtime_lexical_backend",
                "release_required_lexical_backend",
                "production_status",
                "threshold_policy",
                "uncalibrated_behavior",
                "in_memory_backend_scope",
            )
        },
        "persona": {
            key: require_value(persona, key, "persona")
            for key in (
                "profile_type",
                "focus_note_max_words",
                "allowed_fields",
                "reject_role_markers",
                "reject_safety_bypass",
                "reject_diagnosis_or_medication_instruction",
                "reject_crisis_bypass",
            )
        },
        "safety": {
            key: require_value(safety, key, "safety")
            for key in (
                "crisis_interception_before_retrieval",
                "imported_document_is_untrusted_evidence",
                "llm_tool_permissions",
                "buffer_generation_until_final_guard",
                "exact_prompt_visible_citation_binding",
                "grounded_answer_without_valid_citation",
                "web_search_requires_explicit_consent",
            )
        },
        "storage": storage,
        "release_gates": release_gates,
    }


def render_contract(config_path: Path) -> str:
    source = config_path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(source)
    config = require_mapping(loaded, "rag.yaml")
    payload = build_contract(config, sha256_text(source))
    payload["contract_sha256"] = sha256_text(canonical_json(payload))
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = render_contract(args.config)
    if args.write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(expected, encoding="utf-8")
        return 0
    actual = args.output.read_text(encoding="utf-8") if args.output.is_file() else ""
    if actual != expected:
        raise ContractError(
            f"Android RAG runtime contract drifted: regenerate with {args.config} -> {args.output}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        raise SystemExit(f"RAG contract error: {error}")
