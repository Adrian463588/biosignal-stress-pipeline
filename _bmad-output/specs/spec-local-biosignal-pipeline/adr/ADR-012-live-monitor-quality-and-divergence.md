# ADR-012: Live Monitor quality and divergence policy

Status: Accepted (29 July 2026)

## Decision

Live Monitor reads its SQA, inference cadence, baseline, and physiological-arousal divergence thresholds from the versioned Android `live_monitor_config.json` asset. That asset must reference this ADR and accepted model manifests. Model manifests, not UI code, declare the PPG channels required for an inference window.

The initial policy retains the previously implemented thresholds: usable flatline ratio below 0.20, excellent below 0.01, timestamp-gap deviation at most 250 ms with fewer than one second of gaps, minimum quality score 0.50, divergence delta at least 0.20, current stress probability at least 0.65, and a ten-window rolling baseline. Inference cadence is ten seconds.

Missing manifest-declared channels, incomplete windows, rejected SQA, non-monotonic timestamps, unavailable artifacts, or non-accepted manifests produce no prediction or divergence claim. Wearable sensor timestamps are authoritative and must be persisted unchanged.

## Consequences

Changing any threshold or channel requirement requires a replacement ADR and matching configuration/test update. The application remains explicitly unavailable while the release-freeze leaves no `release_status: accepted` manifest in the configured canonical asset path.
