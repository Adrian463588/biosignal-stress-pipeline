"""
trim_raw_data.py
----------------
Pre-wrangling data trimmer.

For each participant, trim raw CSV files based on the exact phase anchor:
  - Folder 'Relaks/' uses the 'Relax' phase window from the anchor CSV.
  - Folder 'Stress/' uses the 'Task' phase window from the anchor CSV.

A configurable BUFFER_SECONDS is added around each phase boundary to avoid
cutting valid edge beats.

Input:  data/raw/DataRenamedPartisipan/
Output: data/raw/DataTrimmedPartisipan/   (new directory, originals untouched)
"""

import logging
import re
import shutil
from datetime import timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ANCHOR_CSV = PROJECT_ROOT / "TimeStampAnchor" / "participants_2026-05-03_22-47_WIB.csv"
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "DataRenamedPartisipan"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "DataTrimmedPartisipan"
MAPPING_CSV = INPUT_DIR / "mapping.csv"

BUFFER_SECONDS: int = 30  # seconds of extra padding around phase boundaries

# Mapping from directory name → anchor phase name in CSV
FOLDER_TO_PHASE: dict[str, str] = {
    "relaks": "Relax",
    "relaksasi": "Relax",  # ID-04 has 'Relaksasi' folder variant
    "stress": "Task",
}

# ---------------------------------------------------------------------------
# Helpers: participant-ID resolution
# ---------------------------------------------------------------------------
_ALIASES: dict[str, str] = {
    "ARYOO05": "ARYO05",
    "RAIIHAN11": "RAIHAN11",
    "DANISH-15": "DANISH15",
}


def _normalize(name: str) -> str:
    """Strip to letters-only uppercase (removes digits and symbols)."""
    return re.sub(r"[^A-Z]", "", name.upper())


def _letters_only(name: str) -> str:
    """Strip to alphanumeric uppercase (used for legacy fallback)."""
    return "".join(ch for ch in name.upper() if ch.isalnum())


def _build_name_to_id_map(mapping_csv: Path) -> dict[str, str]:
    """
    Build {letters_only_name_uppercase -> participant_id} from mapping.csv.

    E.g. 'RAFINDA' -> 'ID-27', 'ALPHONSUS' -> 'ID-26'
    This is the authoritative source of truth for name→ID mapping.
    """
    name_map: dict[str, str] = {}
    if not mapping_csv.exists():
        logger.warning(
            "mapping.csv not found at %s – fallback to suffix-only resolution.",
            mapping_csv,
        )
        return name_map
    df = pd.read_csv(mapping_csv)
    for _, row in df.iterrows():
        pid = str(row.get("participant_id", "")).strip()
        original = str(row.get("original_name", "")).strip()
        if pid and original:
            key = _normalize(original)  # letters only, uppercase
            if key:
                name_map[key] = pid
    return name_map


_NAME_TO_ID: dict[str, str] = {}  # populated in load_phase_windows()


def _code_to_id(raw: str, name_map: dict[str, str] | None = None) -> str:
    """
    Convert anchor participant_code to canonical ID-XX.

    Strategy (in priority order):
    1. Apply known typo aliases.
    2. Exact letters-only match against name_map (mapping.csv).
    3. Prefix match: find the longest name_map key that starts with the
       anchor's letter prefix (handles ALPHONS25 → ALPHONSUS → ID-26).
    4. Fallback: extract trailing number and format as ID-XX.
    """
    raw = _ALIASES.get(raw.upper(), raw)
    # Letters-only prefix (e.g. RAFINDA26 -> RAFINDA, ALPHONS25 -> ALPHONS)
    letters = _normalize(raw)

    if name_map:
        # 1) Exact match
        if letters in name_map:
            return name_map[letters]
        # 2) Prefix match: anchor prefix is a prefix of a full name in map
        #    e.g. ALPHONS starts ALPHONSUS
        #    Require min 5 chars to avoid short ambiguous matches (ALI, etc.)
        if len(letters) >= 5:
            matches = [
                (key, pid)
                for key, pid in name_map.items()
                if key.startswith(letters) or letters.startswith(key)
            ]
            if len(matches) == 1:
                return matches[0][1]
            if len(matches) > 1:
                # Pick the one whose key is the longest prefix of letters
                best = max(matches, key=lambda kv: len(kv[0]))
                logger.debug("Ambiguous prefix match for %r: chose %s", raw, best[1])
                return best[1]

    # Fallback: number suffix
    cleaned = _letters_only(raw)
    m = re.search(r"(\d+)$", cleaned)
    if m:
        return f"ID-{int(m.group(1)):02d}"
    return cleaned


# ---------------------------------------------------------------------------
# Helpers: CSV line detection
# ---------------------------------------------------------------------------


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _is_int(x: str) -> bool:
    try:
        int(float(x))
        return True
    except Exception:
        return False


def _is_ecg_data_line(line: str) -> bool:
    parts = [p.strip() for p in line.split(",")]
    return (
        len(parts) >= 7
        and _is_float(parts[0])
        and _is_int(parts[1])
        and _is_int(parts[2])
    )


def _is_ppg_data_line(line: str) -> bool:
    parts = [p.strip() for p in line.split(",")]
    return len(parts) >= 6 and _is_float(parts[0]) and _is_int(parts[1])


# ---------------------------------------------------------------------------
# Core trim function
# ---------------------------------------------------------------------------


def trim_csv_file(
    input_path: Path,
    output_path: Path,
    min_ms: int,
    max_ms: int,
) -> tuple[int, int]:
    """
    Copy ``input_path`` to ``output_path``, keeping only data rows where
    unix_ms timestamp column is within [min_ms, max_ms].

    Returns (original_data_rows, kept_data_rows).
    """
    is_ecg = "_ECG" in input_path.name.upper()
    detector = _is_ecg_data_line if is_ecg else _is_ppg_data_line
    # ECG: col index 2 is unix_ms; PPG: col index 1 is unix_ms
    time_idx = 2 if is_ecg else 1

    with open(input_path, "r", encoding="utf-8", errors="replace") as fin:
        lines = fin.readlines()

    # Find where data rows start (after header metadata)
    data_start: int | None = None
    for i, line in enumerate(lines):
        if detector(line.strip()):
            data_start = i
            break

    if data_start is None:
        logger.warning(
            "Could not find data start in %s – copying as-is.", input_path.name
        )
        shutil.copy2(input_path, output_path)
        total = len(lines)
        return total, total

    original_count = 0
    kept_count = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        # Preserve metadata headers verbatim
        for i in range(data_start):
            fout.write(lines[i])

        for line in lines[data_start:]:
            stripped = line.strip()
            if not stripped:
                continue
            original_count += 1
            parts = stripped.split(",")
            try:
                ts = int(parts[time_idx].strip())
                if min_ms <= ts <= max_ms:
                    fout.write(line)
                    kept_count += 1
            except (IndexError, ValueError):
                # Malformed line — keep it to avoid silent data loss
                fout.write(line)
                kept_count += 1

    return original_count, kept_count


# ---------------------------------------------------------------------------
# Anchor loading
# ---------------------------------------------------------------------------


def load_phase_windows(
    anchor_csv: Path, buffer_s: int
) -> dict[str, dict[str, tuple[int, int]]]:
    """
    Returns: {participant_id: {phase_name: (min_ms, max_ms)}}

    E.g. {'ID-01': {'Relax': (min_ms, max_ms), 'Task': (min_ms, max_ms)}, ...}

    Uses mapping.csv (same directory as anchor_csv's grandparent INPUT_DIR) as the
    authoritative name→ID lookup to handle cases where the number suffix in the
    anchor code does not match the participant's assigned ID number.
    """
    # Build name→ID map from mapping.csv for accurate resolution
    name_map = _build_name_to_id_map(MAPPING_CSV)
    if name_map:
        logger.info(
            "Loaded name-to-ID map for %d participants from mapping.csv.", len(name_map)
        )
    else:
        logger.warning(
            "Empty name-to-ID map; will fall back to numeric suffix resolution."
        )

    df = pd.read_csv(anchor_csv)
    df["participant_code"] = df["participant_code"].astype(str).str.strip().str.upper()
    df["participant_code"] = df["participant_code"].map(
        lambda c: _code_to_id(c, name_map)
    )
    df["phase"] = df["phase"].astype(str).str.strip()
    df["start_ts"] = pd.to_datetime(df["start_timestamp"], utc=True)
    df["end_ts"] = pd.to_datetime(df["end_timestamp"], utc=True)

    buf = timedelta(seconds=buffer_s)
    windows: dict[str, dict[str, tuple[int, int]]] = {}

    for _, row in df.iterrows():
        pid = row["participant_code"]
        phase = row["phase"]
        if phase not in ("Relax", "Task"):
            continue  # Skip Routine

        # Validate row
        if pd.isna(row["start_ts"]) or pd.isna(row["end_ts"]):
            logger.warning("Skipping invalid anchor row for %s/%s", pid, phase)
            continue
        if row["end_ts"] <= row["start_ts"]:
            logger.warning("Skipping anchor row end<=start for %s/%s", pid, phase)
            continue

        min_ms = int((row["start_ts"] - buf).timestamp() * 1000)
        max_ms = int((row["end_ts"] + buf).timestamp() * 1000)

        windows.setdefault(pid, {})[phase] = (min_ms, max_ms)

    return windows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not INPUT_DIR.exists():
        logger.error("Input directory does not exist: %s", INPUT_DIR)
        return

    phase_windows = load_phase_windows(ANCHOR_CSV, BUFFER_SECONDS)
    logger.info("Loaded anchor windows for %d participants.", len(phase_windows))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if MAPPING_CSV.exists():
        shutil.copy2(MAPPING_CSV, OUTPUT_DIR / "mapping.csv")

    total_files = 0
    total_orig = 0
    total_kept = 0
    skipped_no_anchor = 0

    for participant_dir in sorted(INPUT_DIR.iterdir()):
        if not participant_dir.is_dir():
            continue

        pid = participant_dir.name  # e.g. "ID-31"
        if pid == "mapping.csv" or not pid.startswith("ID-"):
            continue

        out_participant_dir = OUTPUT_DIR / pid
        out_participant_dir.mkdir(exist_ok=True)

        logger.info("Processing %s", pid)

        pid_windows = phase_windows.get(pid)
        if pid_windows is None:
            logger.warning(
                "  No anchor data for %s – copying folder without trimming.", pid
            )
            shutil.copytree(participant_dir, out_participant_dir, dirs_exist_ok=True)
            skipped_no_anchor += 1
            continue

        for condition_dir in sorted(participant_dir.iterdir()):
            if not condition_dir.is_dir():
                continue

            folder_key = condition_dir.name.lower()  # 'relaks' or 'stress'
            phase_name = FOLDER_TO_PHASE.get(folder_key)

            if phase_name is None:
                logger.warning(
                    "  Unknown condition folder '%s' for %s – copying without trimming.",
                    condition_dir.name,
                    pid,
                )
                out_cond_dir = out_participant_dir / condition_dir.name
                shutil.copytree(condition_dir, out_cond_dir, dirs_exist_ok=True)
                continue

            window = pid_windows.get(phase_name)
            if window is None:
                logger.warning(
                    "  No anchor window for %s/%s (phase=%s) – copying without trimming.",
                    pid,
                    condition_dir.name,
                    phase_name,
                )
                out_cond_dir = out_participant_dir / condition_dir.name
                shutil.copytree(condition_dir, out_cond_dir, dirs_exist_ok=True)
                continue

            min_ms, max_ms = window
            out_cond_dir = out_participant_dir / condition_dir.name
            out_cond_dir.mkdir(exist_ok=True)

            logger.info(
                "  %s → phase=%s | window=[%d, %d]",
                condition_dir.name,
                phase_name,
                min_ms,
                max_ms,
            )

            for csv_file in sorted(condition_dir.glob("*.csv")):
                out_csv = out_cond_dir / csv_file.name
                orig, kept = trim_csv_file(csv_file, out_csv, min_ms, max_ms)
                total_files += 1
                total_orig += orig
                total_kept += kept
                pct = (kept / orig * 100) if orig > 0 else 0.0
                logger.info(
                    "    %-55s kept %6d/%6d rows (%5.1f%%)",
                    csv_file.name,
                    kept,
                    orig,
                    pct,
                )

    removed = total_orig - total_kept
    pct_total = (total_kept / total_orig * 100) if total_orig > 0 else 0.0
    logger.info(
        "Done. %d files processed. Kept %d/%d rows (%.1f%%). "
        "Removed %d rows. %d participants skipped (no anchor).",
        total_files,
        total_kept,
        total_orig,
        pct_total,
        removed,
        skipped_no_anchor,
    )


if __name__ == "__main__":
    main()
