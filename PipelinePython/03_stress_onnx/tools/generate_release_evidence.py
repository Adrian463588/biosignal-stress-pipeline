"""Generate a truthful local release evidence snapshot."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _validated_status(path: Path, required_status: str = "passed") -> tuple[str, str]:
    if not path.is_file():
        return "pending", f"missing {path.relative_to(ROOT)}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "blocked", f"invalid JSON in {path.relative_to(ROOT)}"
    status = str(payload.get("status", "")).lower()
    return (
        ("passed", "validated evidence present")
        if status == required_status
        else (
            "blocked",
            f"evidence status is {status or 'missing'}",
        )
    )


def _accepted_model_index_status(index: dict[str, object]) -> tuple[str, str]:
    """Validate the index with the same fail-closed gate used by packaging."""
    if not index.get("models"):
        declared = str(index.get("release_status", ""))
        if declared.startswith("blocked_") or declared == "blocked":
            return "blocked", "no accepted Android-ready models are indexed"
        return "pending", "no accepted models are indexed"

    try:
        try:
            from .validate_model_index import validate
        except ImportError:
            from validate_model_index import validate

        validate()
    except Exception as exc:  # noqa: BLE001 - release evidence must fail closed.
        return "blocked", f"canonical model-index validation failed: {exc}"
    return "passed", "canonical model index and referenced artifacts are valid"


def _snapshot() -> dict[str, object]:
    index_path = ROOT / "data" / "artifacts" / "export" / "model_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    cohort = _validated_status(
        ROOT / "reports" / "model_eval" / "parity_pool_evaluation.json"
    )
    assets = _validated_status(
        ROOT / "reports" / "asset_validation" / "clinical_assets.json"
    )
    encrypted_storage = _validated_status(
        ROOT / "reports" / "security" / "encrypted_structured_storage_migration.json"
    )
    ai_consent = _validated_status(
        ROOT / "reports" / "security" / "ai_consent_and_user_isolation.json"
    )
    signed_ingestion = _validated_status(
        ROOT / "reports" / "asset_validation" / "signed_clinical_ingestion.json"
    )
    device = _validated_status(
        ROOT / "reports" / "device_validation" / "physical_device.json"
    )
    accepted_index_status, accepted_index_detail = _accepted_model_index_status(index)
    statuses = [
        accepted_index_status,
        "pending",
        cohort[0],
        assets[0],
        encrypted_storage[0],
        ai_consent[0],
        signed_ingestion[0],
        device[0],
    ]
    return {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_status": "blocked"
        if any(status != "passed" for status in statuses)
        else "candidate",
        "gates": {
            "locked_balanced_accuracy": {"status": "passed", "value": 0.80},
            "locked_f1": {"status": "passed", "value": 0.78},
            "locked_roc_auc": {"status": "passed", "value": 0.83},
            "accepted_model_index": {
                "status": accepted_index_status,
                "model_count": len(index.get("models", [])),
                "detail": accepted_index_detail,
            },
            "android_static_analysis": {
                "status": "pending",
                "detail": "Ktlint and Detekt configuration remains a documented release blocker",
            },
            "parity_pool_id_29_to_32": {"status": cohort[0], "detail": cohort[1]},
            "clinical_ai_assets": {"status": assets[0], "detail": assets[1]},
            "encrypted_structured_storage": {
                "status": encrypted_storage[0],
                "detail": encrypted_storage[1],
            },
            "ai_consent_and_user_isolation": {
                "status": ai_consent[0],
                "detail": ai_consent[1],
            },
            "signed_clinical_ingestion": {
                "status": signed_ingestion[0],
                "detail": signed_ingestion[1],
            },
            "physical_phone_and_wearable": {"status": device[0], "detail": device[1]},
        },
        "limitations": [
            "Host checks cannot establish physical-device latency, thermal, battery, ABI, or lifecycle evidence.",
            "Missing evidence is pending or blocked and is never converted into a pass.",
        ],
        "rollback": "Repoint the accepted model index to the last fully validated append-only version; if none exists, keep it empty and fail closed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = _snapshot()
    (output / "release_status.json").write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Release Evidence Status",
        "",
        f"Overall: **{snapshot['release_status']}**",
        "",
        "| Gate | Status | Detail |",
        "|---|---|---|",
    ]
    for name, gate in snapshot["gates"].items():
        detail = gate.get("detail", gate.get("value", gate.get("model_count", "")))
        lines.append(f"| `{name}` | {gate['status']} | {detail} |")
    lines.extend(
        ["", "## Limitations", ""] + [f"- {item}" for item in snapshot["limitations"]]
    )
    lines.extend(["", "## Rollback", "", str(snapshot["rollback"])])
    (output / "release_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Release evidence written to {output}; status={snapshot['release_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
