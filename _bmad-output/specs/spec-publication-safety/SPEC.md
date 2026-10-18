---
id: SPEC-publication-safety
companions:
  - acceptance.md
sources: []
---

# Public Repository Safety

## Why

Thesis reviewers and future developers need a reproducible Python pipeline repository. Publication must protect participants and maintainers from disclosure of identities, physiological records, and credentials while keeping the pipeline synchronized with its thesis source.

## Capabilities

- id: CAP-1
  intent: Maintainers can publish reviewed pipeline source without exposing credentials or participant-identifying data.
  success: The publication audit passes for candidate files and all reachable history, and the published tree contains no raw/cleaned records, identity map, or participant names.
- id: CAP-2
  intent: Developers can change the pipeline with repeatable, reviewable quality evidence.
  success: The pull-request CI run passes publication safety, Ruff lint/format, and the selected pytest suite for the proposed revision.

## Constraints

- Participant references use only `ID-01` through `ID-32`; names and name-to-ID maps stay outside Git.
- This repository publishes reviewed source and documentation only; physiological data and model artifacts require a separate explicit release review.
- Pipeline source must be compared with the thesis checkout before publication; unrelated source-worktree changes are excluded.

## Non-goals

- Changing repository visibility or publishing participant datasets, application binaries, or model weights.
- Changing the research, preprocessing, validation, training, or ONNX contracts.

## Success signal

The sanitized source tree matches the reviewed thesis pipeline revision, the local publication audit passes against files and reachable history, and GitHub CI passes on the pushed commit.
