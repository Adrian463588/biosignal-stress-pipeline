#!/usr/bin/env python3
"""Fail-closed publication audit for the code-only public repository."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DATA_SUFFIXES = {
    ".csv",
    ".tsv",
    ".parquet",
    ".feather",
    ".arrow",
    ".h5",
    ".hdf5",
    ".mat",
    ".npy",
    ".npz",
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".onnx",
    ".pkl",
    ".pickle",
    ".joblib",
    ".tflite",
    ".pt",
    ".pth",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".zip",
    ".7z",
    ".rar",
    ".db",
    ".sqlite",
}
SENSITIVE_DIRS = {
    "data",
    "datasets",
    "timestampanchor",
    "personalityanchor",
    "datarawpartisipan",
    "datarawpartisipan-",
    "datarenamedpartisipan",
    "datatrimmedpartisipan",
    "datasyncedecgpartisipan",
    "testdata",
}
SECRET_PATTERNS = {
    "private-key": re.compile(
        rb"-----BEGIN (?:RSA |OPENSSH |EC |DSA |)PRIVATE KEY-----"
    ),
    "aws-access-key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "github-token": re.compile(
        rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"
    ),
    "slack-token": re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "google-api-key": re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b"),
    "stripe-secret": re.compile(rb"\bsk_(?:live|test)_[A-Za-z0-9]{20,}\b"),
    "credential-url": re.compile(
        rb"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?)://[^/\s:@]+:[^@\s/]+@"
    ),
    "bearer-token": re.compile(rb"(?i)\bBearer\s+(?!example\b)[A-Za-z0-9._~+/-]{24,}"),
}
SECRET_ASSIGNMENT = re.compile(
    r"""(?im)\b(?:api[_-]?key|client[_-]?secret|secret[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|password)\b\s*[:=]\s*["']([^"'\r\n]{8,})["']"""
)
PII_VALUE = re.compile(
    r"""(?i)["'](?:participant|subject|person)[ _-]*(?:full[ _-]*)?name["']\s*:\s*["'](?!unknown|redacted|example|placeholder)[^"'\r\n]{2,}["']|["']nama[ _-]*(?:partisipan|peserta)["']\s*:\s*["'][^"'\r\n]{2,}["']"""
)
PLACEHOLDERS = {
    "changeme",
    "change-me",
    "replace-me",
    "your-token",
    "your_api_key",
    "example",
    "placeholder",
    "redacted",
    "dummy",
    "none",
    "null",
}


def git(*args: str, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout if binary else result.stdout.decode("utf-8", errors="replace")


def path_rules(name: str) -> list[str]:
    path = PurePosixPath(name.replace("\\", "/"))
    parts = {part.casefold() for part in path.parts}
    reasons = []
    if parts & SENSITIVE_DIRS or any(
        part.startswith("datarawpartisipan-") for part in parts
    ):
        reasons.append("participant-data-path")
    if path.suffix.casefold() in DATA_SUFFIXES:
        reasons.append("data-or-generated-artifact")
    base = path.name.casefold()
    if base.startswith(".env") and base != ".env.example":
        reasons.append("environment-secret-file")
    if path.suffix.casefold() in {".pem", ".key", ".p12", ".pfx"}:
        reasons.append("private-key-file")
    if base in {"id_rsa", "id_ed25519", "credentials.json", "service-account.json"}:
        reasons.append("credential-file")
    if "mapping.csv" in base:
        reasons.append("identity-mapping")
    return reasons


def content_rules(name: str, content: bytes) -> list[str]:
    reasons = []
    for label, pattern in SECRET_PATTERNS.items():
        if pattern.search(content):
            reasons.append(label)
    text = content.decode("utf-8", errors="ignore")
    for match in SECRET_ASSIGNMENT.finditer(text):
        value = match.group(1).strip().casefold()
        if not any(marker in value for marker in PLACEHOLDERS):
            reasons.append("credential-assignment")
            break
    if PII_VALUE.search(text):
        reasons.append("participant-name-field")
    if name.casefold().endswith(".ipynb"):
        try:
            notebook = json.loads(text)
        except json.JSONDecodeError:
            reasons.append("invalid-notebook")
        else:
            if any(cell.get("outputs") for cell in notebook.get("cells", [])):
                reasons.append("notebook-execution-output")
    if b"\x00" in content and not path_rules(name):
        reasons.append("unclassified-binary-file")
    return reasons


def candidate_paths() -> list[str]:
    result = git(
        "ls-files", "--cached", "--others", "--exclude-standard", "-z", binary=True
    )
    return [
        item.decode("utf-8", errors="replace") for item in result.split(b"\x00") if item
    ]


def scan_worktree() -> tuple[set[tuple[str, str]], int]:
    findings: set[tuple[str, str]] = set()
    count = 0
    for name in candidate_paths():
        path = (ROOT / Path(name)).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            continue
        content = path.read_bytes()
        count += 1
        findings.update((name, reason) for reason in path_rules(name))
        findings.update((name, reason) for reason in content_rules(name, content))
    return findings, count


def scan_history() -> tuple[set[tuple[str, str]], int, int]:
    findings: set[tuple[str, str]] = set()
    commits = git("rev-list", "--all").splitlines()
    scanned_blobs: set[str] = set()
    for commit in commits:
        entries = git("ls-tree", "-rz", "--full-tree", commit, binary=True)
        for entry in entries.split(b"\x00"):
            if not entry:
                continue
            metadata, raw_name = entry.split(b"\t", 1)
            _mode, kind, oid = metadata.decode("ascii").split()
            if kind != "blob":
                continue
            name = raw_name.decode("utf-8", errors="replace")
            findings.update((name, reason) for reason in path_rules(name))
            if oid in scanned_blobs:
                continue
            scanned_blobs.add(oid)
            content = git("cat-file", "blob", oid, binary=True)
            findings.update((name, reason) for reason in content_rules(name, content))
    return findings, len(commits), len(scanned_blobs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        action="store_true",
        help="also scan every commit reachable from local Git refs",
    )
    args = parser.parse_args()
    work_findings, files = scan_worktree()
    findings = set(work_findings)
    commits = blobs = 0
    if args.history:
        history_findings, commits, blobs = scan_history()
        findings.update(history_findings)
    print(
        "Scanned {} candidate files{}.".format(
            files,
            ", {} commits / {} unique historical blobs".format(commits, blobs)
            if args.history
            else "",
        )
    )
    if findings:
        print("Publication audit failed:")
        for name, reason in sorted(findings):
            print("  {}: {}".format(name, reason))
        return 1
    print("Publication audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
