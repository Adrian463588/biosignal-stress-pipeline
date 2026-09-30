"""
batch_rename.py
---------------
Renames participant data directories from real names to anonymised ID-XX format.

Usage (dry-run — no files are changed):
    python PipelinePython/01_data_preprocess/tools/batch_rename.py --dry-run

Usage (execute copy + rename):
    python PipelinePython/01_data_preprocess/tools/batch_rename.py

Behaviour:
  1. Discovers all participant sub-directories inside DATA_ROOT.
  2. Reads the `stoptime` header line from the first PPG CSV found in each
     participant folder to determine the earliest session datetime.
  3. Sorts participants chronologically and assigns IDs: ID-01, ID-02, …
  4. Copies each participant directory to OUTPUT_ROOT / <ID-XX> /.
  5. Renames every CSV inside the copied folder by replacing the original
     participant-name prefix with the new <ID-XX> prefix.
  6. Writes a mapping.csv file (original_name, id, session_datetime) to
     OUTPUT_ROOT for full traceability.

Design:
  - Open/Closed:  new renaming strategies can be added by subclassing Renamer.
  - Single Responsibility: ParticipantDiscovery handles discovery,
    DatetimeParser handles header parsing, Renamer handles the copy+rename.
  - DRY: all path construction is centralised in one place.
  - Non-destructive: original DataRawPartisipan folder is never modified.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Root directory that contains one sub-folder per participant
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# List of raw data folders to process together for global sorting
DATA_ROOTS: list[Path] = [
    PROJECT_ROOT
    / "data"
    / "raw"
    / "DataRawPartisipan-20260505T041918Z-3-001"
    / "DataRawPartisipan",
]

# Destination root: renamed folders will be written here
OUTPUT_ROOT: Path = PROJECT_ROOT / "data" / "raw" / "DataRenamedPartisipan"

# Conditions sub-folders to search for PPG files
CONDITIONS = ("Relaks", "Stress")

# Pattern to identify PPG Green files (used as the reference for session time)
PPG_GREEN_MARKER = "ppg green"

# Header line index (0-based) containing stoptime in PPG CSV files
STOPTIME_LINE_INDEX = 2  # "stoptime=Thu Apr 23 09:57:16 GMT+07:00 2026"
STOPTIME_PREFIX = "stoptime="  # prefix to strip

# Date format embedded in PPG CSV stoptime header
STOPTIME_FORMAT = (
    "%a %b %d %H:%M:%S GMT%z %Y"  # e.g. Thu Apr 23 09:57:16 GMT+07:00 2026
)

# Fallback: parse datetime from filename  e.g. "…_2026-04-23_09-57.csv"
FILENAME_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})")
FILENAME_DATE_FORMAT = "%Y-%m-%d_%H-%M"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ParticipantInfo:
    """Holds discovery metadata for one participant."""

    original_name: str
    directory: Path
    session_datetime: datetime
    participant_id: str = ""  # filled in after sorting


# ---------------------------------------------------------------------------
# Discovery & parsing (Single Responsibility)
# ---------------------------------------------------------------------------


class DatetimeParser:
    """Extracts session datetime from a PPG CSV file header or filename."""

    @staticmethod
    def from_csv_header(csv_path: Path) -> datetime | None:
        """Read stoptime from PPG CSV header (line 3, 0-indexed line 2)."""
        try:
            with open(csv_path, encoding="utf-8", errors="replace") as fh:
                lines = [fh.readline() for _ in range(STOPTIME_LINE_INDEX + 1)]
            raw = lines[STOPTIME_LINE_INDEX].strip()
            if raw.startswith(STOPTIME_PREFIX):
                value = raw[len(STOPTIME_PREFIX) :]
                # Normalise GMT+07:00 -> +07:00 so strptime can parse it
                value = value.replace("GMT", "")
                return datetime.strptime(value.strip(), "%a %b %d %H:%M:%S %z %Y")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Header parse failed for %s: %s", csv_path.name, exc)
        return None

    @staticmethod
    def from_filename(csv_path: Path) -> datetime | None:
        """Fallback: extract datetime from filename e.g. '…_2026-04-23_09-57.csv'."""
        match = FILENAME_DATE_PATTERN.search(csv_path.stem)
        if match:
            try:
                return datetime.strptime(match.group(1), FILENAME_DATE_FORMAT).replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                pass
        return None

    @classmethod
    def parse(cls, csv_path: Path) -> datetime | None:
        """Try header first, then filename fallback."""
        return cls.from_csv_header(csv_path) or cls.from_filename(csv_path)


class ParticipantDiscovery:
    """Traverses DATA_ROOT and collects participant metadata."""

    def __init__(self, data_roots: list[Path] = DATA_ROOTS) -> None:
        self._roots = data_roots

    def discover(self, start_id: int = 1) -> list[ParticipantInfo]:
        """
        Return a list of ParticipantInfo sorted by session_datetime (ascending).
        Participants whose session datetime cannot be determined are placed last.
        """
        participants: list[ParticipantInfo] = []

        for root in self._roots:
            if not root.exists():
                logger.warning("Data root does not exist: %s", root)
                continue

            for participant_dir in sorted(root.iterdir()):
                if not participant_dir.is_dir():
                    continue

                session_dt = self._earliest_session_datetime(participant_dir)
                participants.append(
                    ParticipantInfo(
                        original_name=participant_dir.name,
                        directory=participant_dir,
                        session_datetime=session_dt
                        or datetime.max.replace(tzinfo=timezone.utc),
                    )
                )

        participants.sort(key=lambda p: p.session_datetime)

        # Assign ID-XX after sorting
        for idx, info in enumerate(participants, start=start_id):
            info.participant_id = f"ID-{idx:02d}"

        return participants

    @staticmethod
    def _earliest_session_datetime(participant_dir: Path) -> datetime | None:
        """Find the earliest session datetime across all conditions."""
        datetimes: list[datetime] = []
        for condition in CONDITIONS:
            condition_dir = participant_dir / condition
            if not condition_dir.is_dir():
                continue
            for csv_file in condition_dir.glob("*.csv"):
                if PPG_GREEN_MARKER in csv_file.name.lower():
                    dt = DatetimeParser.parse(csv_file)
                    if dt:
                        datetimes.append(dt)
                    break  # one PPG green file per condition is enough
        return min(datetimes) if datetimes else None


# ---------------------------------------------------------------------------
# Renaming engine (Single Responsibility)
# ---------------------------------------------------------------------------


class Renamer:
    """
    Copies a participant directory to the output root under the new ID name
    and renames all CSV files inside by substituting the participant's original
    name prefix with the new ID prefix.
    """

    def __init__(self, output_root: Path = OUTPUT_ROOT, dry_run: bool = False) -> None:
        self._output_root = output_root
        self._dry_run = dry_run

    def rename_all(self, participants: list[ParticipantInfo]) -> None:
        """Process every participant in the list."""
        if not self._dry_run:
            self._output_root.mkdir(parents=True, exist_ok=True)

        for info in participants:
            self._rename_participant(info)

        self._write_mapping(participants)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _rename_participant(self, info: ParticipantInfo) -> None:
        dest_dir = self._output_root / info.participant_id
        action = "[DRY-RUN] Would copy" if self._dry_run else "Copying"
        logger.info(
            "%s  '%s'  ->  '%s'",
            action,
            info.original_name,
            info.participant_id,
        )

        if not self._dry_run:
            if dest_dir.exists():
                logger.warning("Destination already exists, skipping: %s", dest_dir)
            else:
                shutil.copytree(info.directory, dest_dir)

        self._rename_csvs_in(dest_dir, info.original_name, info.participant_id)

    def _rename_csvs_in(self, base_dir: Path, original_name: str, new_id: str) -> None:
        """
        Recursively rename every CSV under base_dir whose filename starts with
        original_name (case-insensitive) by replacing that prefix with new_id.

        Example:
          AbelRelaks14-PPG Green-2026-04-23_09-50.csv
          ->  ID-01Relaks14-PPG Green-2026-04-23_09-50.csv
        """
        if self._dry_run:
            # Operate on source directory for dry-run preview
            base_dir = (
                self._output_root / new_id
                if False  # keep reference to original
                else base_dir.parent.parent / original_name
                if not base_dir.exists()
                else base_dir
            )

        search_root = base_dir
        if self._dry_run:
            # For dry-run, we need to find which raw folder the original came from
            for root in DATA_ROOTS:
                potential_source = root / original_name
                if potential_source.exists():
                    search_root = potential_source
                    break

        # Pattern to find 'Relaks' or 'Stress' (or 'Stres') in the filename
        # We replace everything before the condition keyword with the new ID,
        # and strip any digits immediately following the condition.
        pattern = re.compile(r"^(.*?)(Relaks|Stress|Stres)\d*(.*)", re.IGNORECASE)

        for csv_path in search_root.rglob("*.csv"):
            match = pattern.match(csv_path.name)
            if not match:
                logger.debug(
                    "  Skipping file (no condition keyword found): %s", csv_path.name
                )
                continue

            condition_str = match.group(2).capitalize()
            if condition_str == "Stres":
                condition_str = "Stress"

            new_name = new_id + condition_str + match.group(3)
            new_path = csv_path.parent / new_name

            if self._dry_run:
                logger.info(
                    "  [DRY-RUN] Would rename  '%s'  ->  '%s'",
                    csv_path.name,
                    new_name,
                )
            else:
                logger.info("  Renaming  '%s'  ->  '%s'", csv_path.name, new_name)
                csv_path.rename(new_path)

    def _write_mapping(self, participants: list[ParticipantInfo]) -> None:
        """Write mapping.csv to output root for traceability."""
        mapping_path = self._output_root / "mapping.csv"
        rows = [
            {
                "participant_id": p.participant_id,
                "original_name": p.original_name,
                "session_datetime": (
                    p.session_datetime.isoformat()
                    if p.session_datetime != datetime.max.replace(tzinfo=timezone.utc)
                    else "unknown"
                ),
            }
            for p in participants
        ]

        if self._dry_run:
            logger.info("[DRY-RUN] mapping.csv would append:")
            for r in rows:
                logger.info("  %s", r)
            return

        self._output_root.mkdir(parents=True, exist_ok=True)
        # Always overwrite mapping.csv for global run
        with open(mapping_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["participant_id", "original_name", "session_datetime"]
            )
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Mapping written to: %s", mapping_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rename participant data folders to anonymised ID-XX format. "
            "By default, copies data to DataRenamedPartisipan/ without "
            "touching the originals."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing anything to disk.",
    )
    parser.add_argument(
        "--data-roots",
        nargs="+",
        type=Path,
        default=DATA_ROOTS,
        help="List of raw data directories to process together.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Override the output directory. Default: {OUTPUT_ROOT}",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    start_id = 1
    discovery = ParticipantDiscovery(data_roots=args.data_roots)
    participants = discovery.discover(start_id=start_id)

    if not participants:
        logger.error("No participant directories found in provided roots.")
        return

    logger.info("Discovered %d participant(s):", len(participants))
    for p in participants:
        logger.info(
            "  %-20s  ->  %s  (session: %s)",
            p.original_name,
            p.participant_id,
            p.session_datetime.isoformat()
            if p.session_datetime != datetime.max.replace(tzinfo=timezone.utc)
            else "unknown",
        )

    renamer = Renamer(output_root=args.output_root, dry_run=args.dry_run)
    renamer.rename_all(participants)

    if args.dry_run:
        logger.info("Dry-run complete. No files were modified.")
    else:
        logger.info("Done. Renamed data written to: %s", args.output_root)


if __name__ == "__main__":
    main()
