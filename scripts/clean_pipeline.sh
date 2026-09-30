#!/usr/bin/env bash
# Clean slate script for Biosignal pipeline
# As per PRD and AGENTS.md requirements

set -e

BACKUP=false
if [ "$1" == "--backup" ]; then
    BACKUP=true
fi

if [ "$BACKUP" = true ]; then
    echo "Creating backup of artifacts..."
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    mkdir -p backups/artifacts_$TIMESTAMP
    cp -r data/artifacts/* backups/artifacts_$TIMESTAMP/ || true
    echo "Backup saved to backups/artifacts_$TIMESTAMP/"
fi

echo "Cleaning artifacts..."
rm -rf data/artifacts/bronze/* || true
rm -rf data/artifacts/silver/* || true
rm -rf data/artifacts/gold/* || true
rm -rf data/artifacts/models/* || true
rm -rf data/artifacts/export/* || true
rm -rf reports/eda/* || true
rm -rf reports/sqa/* || true
rm -rf reports/validation/* || true
rm -rf reports/model_eval/* || true
rm -rf reports/audit/* || true
rm -rf data/gold/waveforms/* || true
rm -rf mlruns/* || true

echo "Cleanup complete."
