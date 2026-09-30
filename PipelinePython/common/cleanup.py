"""Recoverable cleanup for generated pipeline artifacts.

The cleanup contract is deliberately allowlist-only. Raw inputs and the four
legacy canonical artifacts are never eligible for movement or deletion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


PROTECTED_RELATIVE_PATHS: tuple[Path, ...] = (
    Path("data/artifacts/bronze/canonical_signals.csv"),
    Path("data/artifacts/silver/canonical_segmented_signals.csv"),
    Path("data/artifacts/gold/gold_features.csv"),
    Path("data/artifacts/validation/channel_validation_metrics.csv"),
)

GENERATED_ROOTS: tuple[Path, ...] = (
    Path("DataGenerated"),
    Path("models"),
    Path("data/bronze"),
    Path("data/sync"),
    Path("data/segmented"),
    Path("data/silver"),
    Path("data/gold"),
    Path("reports"),
    Path("data/artifacts/eda"),
    Path("data/artifacts/models"),
    Path("data/artifacts/export"),
    Path("data/artifacts/mlruns"),
    Path("data/artifacts/mlflow"),
    Path("data/artifacts/validation_synced_ecg"),
    Path("android/mobile/build"),
    Path("android/shared/build"),
    Path("android/wear/build"),
    Path("android/inference/build"),
    Path("android/mobile/src/main/assets/models"),
)

PARTIAL_GENERATED_DIRS: tuple[Path, ...] = (
    Path("data/artifacts/bronze"),
    Path("data/artifacts/silver"),
    Path("data/artifacts/gold"),
    Path("data/artifacts/validation"),
)

ANDROID_GENERATED_ASSET_PATTERNS: tuple[str, ...] = (
    "stress_*_model.onnx",
    "stress_*_fp32.onnx",
    "stress_*_int8.onnx",
    "selected_features_*.json",
    "model_manifest*.json",
    "test_vectors.json",
)


@dataclass(frozen=True)
class CleanupEntry:
    relative_path: str
    kind: str
    file_count: int
    size_bytes: int
    sha256: str | None


@dataclass(frozen=True)
class CleanupResult:
    project_root: str
    backup_root: str
    created_at: str
    applied: bool
    baseline_commit: str
    git_status: str
    protected_hashes: dict[str, str]
    entries: tuple[CleanupEntry, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_stats(path: Path) -> tuple[int, int, str | None]:
    if path.is_file():
        return 1, path.stat().st_size, _sha256(path)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    return len(files), sum(item.stat().st_size for item in files), None


def _git_output(project_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip() if completed.returncode == 0 else "NO_VCS"


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_roots(project_root: Path, backup_root: Path) -> tuple[Path, Path]:
    project = project_root.resolve(strict=True)
    backup = backup_root.resolve(strict=False)
    if backup == project or _is_within(backup, project):
        raise ValueError("backup_root must be outside the project root")
    raw_root = (project / "data/raw").resolve(strict=True)
    if not _is_within(raw_root, project):
        raise ValueError("raw data root escaped the project root")
    return project, backup


def _protected_paths(project_root: Path) -> set[Path]:
    return {
        (project_root / relative).resolve(strict=False)
        for relative in PROTECTED_RELATIVE_PATHS
    }


def _iter_cleanup_targets(project_root: Path) -> Iterable[Path]:
    protected = _protected_paths(project_root)
    seen: set[Path] = set()

    for relative in GENERATED_ROOTS:
        candidate = (project_root / relative).resolve(strict=False)
        if candidate.exists() and candidate not in seen:
            seen.add(candidate)
            yield candidate

    for relative in PARTIAL_GENERATED_DIRS:
        parent = (project_root / relative).resolve(strict=False)
        if not parent.is_dir():
            continue
        for child in sorted(parent.iterdir()):
            resolved = child.resolve(strict=False)
            if resolved in protected or resolved in seen:
                continue
            seen.add(resolved)
            yield resolved

    assets = project_root / "android/mobile/src/main/assets"
    if assets.is_dir():
        for pattern in ANDROID_GENERATED_ASSET_PATTERNS:
            for child in sorted(assets.glob(pattern)):
                resolved = child.resolve(strict=False)
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved


def _hash_protected_inputs(project_root: Path) -> dict[str, str]:
    roots: Sequence[Path] = (
        project_root / "data/raw",
        project_root / "TimeStampAnchor",
        project_root / "PersonalityAnchor",
    )
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(item for item in root.rglob("*") if item.is_file())
    files.extend(
        path
        for relative in PROTECTED_RELATIVE_PATHS
        if (path := project_root / relative).is_file()
    )
    return {
        path.relative_to(project_root).as_posix(): _sha256(path)
        for path in sorted(set(files))
    }


def build_cleanup_plan(project_root: Path, backup_root: Path) -> CleanupResult:
    project, backup = _validate_roots(project_root, backup_root)
    entries: list[CleanupEntry] = []
    for path in _iter_cleanup_targets(project):
        if not _is_within(path, project):
            raise ValueError(f"cleanup target escaped project root: {path}")
        if _is_within(path, project / "data/raw"):
            raise ValueError(f"raw data cannot be cleaned: {path}")
        count, size, digest = _tree_stats(path)
        entries.append(
            CleanupEntry(
                relative_path=path.relative_to(project).as_posix(),
                kind="directory" if path.is_dir() else "file",
                file_count=count,
                size_bytes=size,
                sha256=digest,
            )
        )
    return CleanupResult(
        project_root=str(project),
        backup_root=str(backup),
        created_at=datetime.now().astimezone().isoformat(),
        applied=False,
        baseline_commit=_git_output(project, "rev-parse", "HEAD"),
        git_status=_git_output(project, "status", "--short"),
        protected_hashes=_hash_protected_inputs(project),
        entries=tuple(entries),
    )


def apply_cleanup(plan: CleanupResult) -> CleanupResult:
    project, backup = _validate_roots(Path(plan.project_root), Path(plan.backup_root))
    if backup.exists() and any(backup.iterdir()):
        raise FileExistsError(f"backup root must be empty: {backup}")
    backup.mkdir(parents=True, exist_ok=True)

    planned_paths = {entry.relative_path for entry in plan.entries}
    current_paths = {
        path.relative_to(project).as_posix() for path in _iter_cleanup_targets(project)
    }
    if current_paths != planned_paths:
        raise RuntimeError("cleanup targets changed after dry-run; rebuild the plan")

    for entry in plan.entries:
        source = (project / entry.relative_path).resolve(strict=True)
        if not _is_within(source, project) or _is_within(source, project / "data/raw"):
            raise ValueError(f"unsafe cleanup source: {source}")
        destination = backup / entry.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

    applied = CleanupResult(
        **{**asdict(plan), "applied": True, "entries": plan.entries}
    )
    manifest_path = backup / "cleanup_manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(applied), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    reference_path = project / "reports" / "migration" / "cleanup_reference_v3.json"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(
        json.dumps(
            {
                "cleanup_manifest": str(manifest_path),
                "cleanup_manifest_sha256": _sha256(manifest_path),
                "protected_file_count": len(applied.protected_hashes),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return applied


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plan = build_cleanup_plan(args.project_root, args.backup_root)
    result = apply_cleanup(plan) if args.apply else plan
    print(
        json.dumps(
            {
                "applied": result.applied,
                "backup_root": result.backup_root,
                "targets": len(result.entries),
                "files": sum(entry.file_count for entry in result.entries),
                "bytes": sum(entry.size_bytes for entry in result.entries),
                "protected_files": len(result.protected_hashes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
