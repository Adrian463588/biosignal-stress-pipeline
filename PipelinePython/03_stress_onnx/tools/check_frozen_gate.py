"""Fail-closed validation for the locked tiered release policy and ADR-aware edits."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
GATE_FILE = ROOT / "PipelinePython" / "configs" / "frozen_gates.yaml"
ADR_DIR = ROOT / "_bmad-output" / "specs" / "spec-local-biosignal-pipeline" / "adr"
MINIMUM_RESEARCH_GATE = (Decimal("0.70"), Decimal("0.70"), Decimal("0.70"))
PREFERRED_GATE = (Decimal("0.80"), Decimal("0.78"), Decimal("0.83"))


def _section_gate(section: str, name: str) -> Decimal:
    text = GATE_FILE.read_text(encoding="utf-8")
    section_match = re.search(
        rf"^  {re.escape(section)}:\s*$([\s\S]*?)(?=^  \S|\Z)",
        text,
        re.MULTILINE,
    )
    if section_match is None:
        raise RuntimeError(f"frozen_gates.yaml must define {section}")
    matches = re.findall(
        rf"^    {re.escape(name)}\s*:\s*([^\s#]+)",
        section_match.group(1),
        re.MULTILINE,
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"frozen_gates.yaml must define {section}.{name} exactly once"
        )
    try:
        return Decimal(matches[0])
    except InvalidOperation as exc:
        raise RuntimeError(f"{section}.{name} must be numeric") from exc


def _gate_changed(base_ref: str | None) -> bool:
    if not base_ref:
        return False
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            "PipelinePython/configs/frozen_gates.yaml",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        return True
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            base_ref,
            "--",
            "PipelinePython/configs/frozen_gates.yaml",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _approved_adr_references(text: str) -> list[str]:
    approved: list[str] = []
    for adr_id in sorted(set(re.findall(r"\bADR-\d{3}\b", text, re.IGNORECASE))):
        candidates = list(ADR_DIR.glob(f"{adr_id.upper()}-*.md"))
        if len(candidates) != 1:
            continue
        content = candidates[0].read_text(encoding="utf-8")
        if re.search(
            r"^## Status\s*(?:\r?\n)+\s*Accepted\b",
            content,
            re.MULTILINE | re.IGNORECASE,
        ):
            approved.append(adr_id.upper())
    return approved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ref", help="Local git baseline, normally HEAD")
    parser.add_argument(
        "--adr-reference",
        action="append",
        default=[],
        help="Accepted ADR authorizing a deliberate frozen-gate file edit",
    )
    args = parser.parse_args()

    policy_text = GATE_FILE.read_text(encoding="utf-8")
    if "version: ml_dl_tiered_v1" not in policy_text:
        raise RuntimeError(
            "frozen_gates.yaml must keep performance policy version ml_dl_tiered_v1"
        )
    minimum = tuple(
        _section_gate("minimum_research", name)
        for name in ("balanced_accuracy", "f1", "roc_auc")
    )
    preferred = tuple(
        _section_gate("preferred", name)
        for name in ("balanced_accuracy", "f1", "roc_auc")
    )
    if minimum != MINIMUM_RESEARCH_GATE:
        raise RuntimeError(
            "minimum research gate must remain exactly 0.70/0.70/0.70; "
            "a model failure must never be repaired by weakening the gate."
        )
    if preferred != PREFERRED_GATE:
        raise RuntimeError("preferred gate must remain exactly 0.80/0.78/0.83")

    if _gate_changed(args.base_ref):
        approved = _approved_adr_references(" ".join(args.adr_reference))
        if "ADR-015" not in approved:
            raise RuntimeError(
                "PipelinePython/configs/frozen_gates.yaml differs from the local baseline, but no "
                "--adr-reference names accepted ADR-015"
            )
        print(f"Gate file edit authorized by: {', '.join(approved)}")

    print(
        "Locked tiered policy verified: research 0.70/0.70/0.70; preferred 0.80/0.78/0.83"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"GATE CHECK FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
