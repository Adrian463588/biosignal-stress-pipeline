"""Repair the checked Android bundle metadata after the fractional-rate fix.

This command is intentionally narrow: it never changes ONNX bytes, model
thresholds, labels, or device receipts.  It adds the manifest-owned filter
contract and explicit source/model/observed-rate metadata to the current
candidate and production vector bundles, refreshes content descriptors, and
records the pre-fix checksums in a quarantine report.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "data" / "artifacts" / "export"
VERSION = "stress_v4.0.0-core240s-20260810T031618Z"
SOURCE_RATE_HZ = {"ppg_green": 25.0, "ppg_ir": 100.0, "ppg_red": 100.0}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filter_contract(channel: str) -> dict[str, Any]:
    high = 5.0 if channel == "ppg_green" else 8.0
    values: dict[str, Any] = {
        "algorithm": "butterworth_sosfiltfilt",
        "order": 4,
        "low_cut_hz": 0.5,
        "high_cut_hz": high,
        "zero_phase": True,
        "pad_policy": "scipy_odd_sosfiltfilt_default",
        "design_version": "sos_zpk_bilinear_v1",
    }

    def canonical_number(value: float) -> str:
        return f"{value:.15f}".rstrip("0").rstrip(".")

    canonical = "\n".join(
        (
            values["algorithm"],
            str(values["order"]),
            canonical_number(values["low_cut_hz"]),
            canonical_number(values["high_cut_hz"]),
            str(values["zero_phase"]).lower(),
            values["pad_policy"],
            values["design_version"],
        )
    )
    values["fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    return values


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def manifest_digest(manifest: dict[str, Any]) -> str:
    canonical = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def android_contract_digest(manifest: dict[str, Any]) -> str:
    payload = "\n".join(
        (
            str(manifest["model_version"]),
            str(manifest["preprocessing_version"]),
            str(manifest["feature_version"]),
            str(manifest["sensor_channel"]),
            str(int(manifest["analysis_seconds"])),
            str(int(manifest["window_seconds"])),
            str(int(manifest["stride_seconds"])),
            str(int(manifest["expected_windows_per_session"])),
            repr(float(manifest["source_rate_hz"])),
            repr(float(manifest["model_rate_hz"])),
            json.dumps(
                manifest.get("filter_contract", {}),
                sort_keys=True,
                separators=(",", ":"),
            ),
            str(bool(manifest["android_ready"])),
            ",".join(str(name) for name in manifest["feature_order"]),
            str(float(manifest["probability_threshold"])),
            str(manifest["onnx_sha256"]),
            str(manifest["onnx_input"]["name"]),
            str(manifest["onnx_output"]["name"]),
            str(manifest.get("performance_policy_version", "")),
            str(manifest.get("performance_policy_sha256", "")),
            str(manifest.get("performance_tier", "")),
            str(manifest.get("deployment_scope", "")),
            str(manifest.get("research_disclaimer", "")),
            str(manifest.get("device_parity_receipt_sha256", "")),
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def update_manifest(path: Path, contract: dict[str, Any]) -> tuple[str, str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    old = sha256(path)
    manifest["filter_contract"] = contract
    channel = str(manifest.get("sensor_channel", ""))
    if channel not in SOURCE_RATE_HZ:
        raise ValueError(f"unsupported PPG channel in manifest: {channel}")
    # The source contract is the nominal PDSatu individual-tracker rate.
    # Timestamp-derived cadence remains a separate diagnostic field.
    manifest["source_rate_hz"] = SOURCE_RATE_HZ[channel]
    manifest["filter_algorithm"] = contract["algorithm"]
    manifest["filter_order"] = contract["order"]
    manifest["filter_low_cut_hz"] = contract["low_cut_hz"]
    manifest["filter_high_cut_hz"] = contract["high_cut_hz"]
    manifest["filter_zero_phase"] = contract["zero_phase"]
    manifest["filter_pad_policy"] = contract["pad_policy"]
    manifest["filter_design_version"] = contract["design_version"]
    manifest["filter_contract_fingerprint"] = contract["fingerprint"]
    if "android_contract_sha256" in manifest:
        manifest["android_contract_sha256"] = android_contract_digest(manifest)
    manifest["manifest_sha256"] = manifest_digest(manifest)
    write_json(path, manifest)
    return old, sha256(path)


def update_vectors(
    path: Path, manifest: dict[str, Any], contract: dict[str, Any]
) -> tuple[str, str]:
    old = sha256(path)
    vectors = json.loads(path.read_text(encoding="utf-8"))
    vectors["vector_schema_version"] = "2.0.0"
    vectors["source_rate_hz"] = manifest["source_rate_hz"]
    vectors["model_rate_hz"] = manifest["model_rate_hz"]
    vectors["filter_contract"] = contract
    for case in vectors.get("cases", []):
        if case.get("kind") != "valid_raw_window":
            continue
        # These checked-in vectors were generated from nominal PDSatu callback
        # timelines; preserve that observed diagnostic while moving the model
        # target to the manifest's fractional rate.
        old_rate = SOURCE_RATE_HZ[manifest["sensor_channel"]]
        case["source_rate_hz"] = manifest["source_rate_hz"]
        case["model_rate_hz"] = manifest["model_rate_hz"]
        case["observed_rate_hz"] = old_rate
        case["sampling_rate_hz"] = manifest["model_rate_hz"]
        case["filter_contract"] = contract
    write_json(path, vectors)
    return old, sha256(path)


def refresh_index(path: Path, replacements: dict[str, str]) -> None:
    index = json.loads(path.read_text(encoding="utf-8"))
    for entry in index.get("models", []):
        channel = str(entry.get("sensor_channel", ""))
        manifest_descriptor = entry.get("files", {}).get("manifest", {})
        manifest_path = EXPORT / str(manifest_descriptor.get("path", ""))
        if channel in SOURCE_RATE_HZ and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry["source_rate_hz"] = manifest["source_rate_hz"]
            entry["model_rate_hz"] = manifest["model_rate_hz"]
        for descriptor in entry.get("files", {}).values():
            descriptor_path = descriptor.get("path")
            if descriptor_path in replacements:
                descriptor["sha256"] = replacements[descriptor_path]
    write_json(path, index)


def main() -> None:
    old_checksums: list[dict[str, str]] = []
    replacements: dict[str, str] = {}
    for channel in ("ppg_green", "ppg_ir"):
        base = EXPORT / VERSION / channel
        contract = filter_contract(channel)
        candidate_manifest_path = base / "model_manifest.json"
        production_manifest_path = base / "production_model_manifest.json"
        candidate_old, candidate_new = update_manifest(
            candidate_manifest_path, contract
        )
        production_before = json.loads(
            production_manifest_path.read_text(encoding="utf-8")
        )
        if "source_candidate_manifest_sha256" in production_before:
            production_before["source_candidate_manifest_sha256"] = candidate_new
            write_json(production_manifest_path, production_before)
        production_old, production_new = update_manifest(
            production_manifest_path, contract
        )
        candidate_manifest = json.loads(
            candidate_manifest_path.read_text(encoding="utf-8")
        )
        production_manifest = json.loads(
            production_manifest_path.read_text(encoding="utf-8")
        )
        for path, manifest in (
            (base / "test_vectors.json", candidate_manifest),
            (base / "production_test_vectors.json", production_manifest),
        ):
            old, new = update_vectors(path, manifest, contract)
            replacements[path.relative_to(EXPORT).as_posix()] = new
            old_checksums.append(
                {"path": path.relative_to(EXPORT).as_posix(), "sha256": old}
            )
        replacements[candidate_manifest_path.relative_to(EXPORT).as_posix()] = (
            candidate_new
        )
        replacements[production_manifest_path.relative_to(EXPORT).as_posix()] = (
            production_new
        )
        old_checksums.extend(
            [
                {
                    "path": candidate_manifest_path.relative_to(EXPORT).as_posix(),
                    "sha256": candidate_old,
                },
                {
                    "path": production_manifest_path.relative_to(EXPORT).as_posix(),
                    "sha256": production_old,
                },
            ]
        )
    refresh_index(EXPORT / "model_index.json", replacements)
    refresh_index(EXPORT / "candidate_model_index.json", replacements)
    quarantine = EXPORT / "quarantine" / "fractional_filter_contract_20260813.json"
    write_json(
        quarantine,
        {
            "reason": "legacy vectors used ambiguous nominal sampling_rate_hz metadata",
            "replacement_contract": "sos_zpk_bilinear_v1",
            "old_artifacts": old_checksums,
        },
    )
    print(
        json.dumps(
            {"updated": sorted(replacements), "quarantine": str(quarantine)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
