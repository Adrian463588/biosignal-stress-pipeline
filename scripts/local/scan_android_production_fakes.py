"""Reject known fake/simulated clinical behavior from active Android sources locally."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = [
    ROOT / "android" / module / "src" / "main"
    for module in (
        "app",
        "core",
        "data",
        "domain",
        "feature-counselor",
        "feature-history",
        "feature-home",
        "feature-live",
        "feature-profile",
        "feature-verbatim",
        "inference",
        "mobile",
        "shared",
        "wear",
    )
]
LEGACY_EXCLUDED_PREFIXES = (
    Path("android/app/src/main/java/com/stressdetector/mobile/data/verbatim"),
    Path("android/app/src/main/java/com/stressdetector/mobile/domain/verbatim"),
    Path("android/app/src/main/java/com/stressdetector/mobile/presentation"),
    Path("android/mobile/src/main/java/com/stressdetector/mobile/data/verbatim"),
    Path("android/mobile/src/main/java/com/stressdetector/mobile/domain/verbatim"),
    Path("android/mobile/src/main/java/com/stressdetector/mobile/presentation"),
)
FORBIDDEN = re.compile(
    r"mock generated|mock transcript|mock retrieval|mocking (?:the )?audio|"
    r"mocking (?:the )?diarization|mocking transcription|mock vad|mock verification|"
    r"simulatedconfidence|fallbackmocktext|mockembedding|active_session_placeholder|"
    r"patient_signature_mock|fabricated (?:dialogue|speaker)|zero[- ]vector fallback|no-op tts",
    re.IGNORECASE,
)


def main() -> int:
    findings: list[str] = []
    for source_root in SOURCE_ROOTS:
        for path in sorted(source_root.rglob("*")):
            if path.suffix.lower() not in {".kt", ".java"}:
                continue
            relative = path.relative_to(ROOT)
            if any(
                relative.is_relative_to(prefix) for prefix in LEGACY_EXCLUDED_PREFIXES
            ):
                continue
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if FORBIDDEN.search(line):
                    findings.append(f"{relative}:{number}: {line.strip()}")
    if findings:
        print("Production fake scan failed:\n" + "\n".join(findings), file=sys.stderr)
        return 1
    print(
        "Active Android production sources contain no forbidden fake/simulation markers"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
