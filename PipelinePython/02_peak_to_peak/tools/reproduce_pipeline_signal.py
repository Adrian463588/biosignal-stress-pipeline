"""Run isolated PipelineSignal reproduction without touching canonical workflows."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PipelinePython.common.legacy_reproduction import (  # noqa: E402
    LegacyReproductionRunner,
)

RUN_ID_PATTERN = re.compile(r"\d{8}T\d{6}Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnostic-only reproduction of the PipelineSignal estimand."
    )
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    run_id = arguments.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must use UTC format YYYYMMDDTHHMMSSZ")
    if not RUN_ID_PATTERN.fullmatch(arguments.source_run_id):
        raise ValueError("source_run_id must use UTC format YYYYMMDDTHHMMSSZ")
    result = LegacyReproductionRunner(
        PROJECT_ROOT,
        arguments.source_run_id,
        run_id,
    ).run()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
