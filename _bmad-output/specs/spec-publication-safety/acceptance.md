# Publication Safety Acceptance (BDD)

## Feature: Publish reviewed pipeline source safely

### Scenario: Reviewed code and documentation are publishable

```gherkin
Given the candidate repository contains reviewed source and documentation, with participant references represented only as ID-01 through ID-32
When the maintainer runs `python scripts/audit_public_repo.py --history`
Then the audit passes without physiological records, identity mappings, credential patterns, or executed notebook output
```

### Scenario: Sensitive files or values block publication

```gherkin
Given a candidate file or reachable commit contains a prohibited data artifact, credential pattern, participant-name field, or notebook execution output
When the publication audit runs
Then it exits with failure and reports only the path and rule category, never the matched secret value
```

### Scenario: Pipeline source stays aligned with the thesis checkout

```gherkin
Given a reviewed thesis source revision and a clean publication clone
When the maintainer synchronizes the pipeline for publication
Then every tracked `PipelinePython/` file matches that source revision and unrelated data or worktree files remain excluded
```
