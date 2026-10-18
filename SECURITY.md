# Security Policy

## Reporting a suspected exposure

Do not create a public issue containing a credential, participant name, identity mapping, raw or cleaned biosignal, or participant-level report. Use GitHub's private vulnerability reporting flow (**Security → Report a vulnerability**) when it is available. If it is unavailable, contact the repository owner through their GitHub profile and identify only the affected path and commit; do not include the sensitive value or data.

If a credential may have been exposed, revoke or rotate it with its issuer before investigating history. Removing a value from the latest commit does not invalidate a credential or erase earlier copies.

## Repository policy

- Only reviewed source code and documentation belong in this repository.
- Participant references in code and documentation use `ID-01` through `ID-32`; never publish names or a name-to-ID mapping.
- Keep raw/cleaned recordings, anchors, participant-level tables, model weights, and generated reports outside Git.
- Run `python scripts/audit_public_repo.py --history` before pushing. CI repeats the audit against candidate files and reachable history.
