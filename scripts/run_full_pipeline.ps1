param([string]$EvidenceDirectory)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'local\LocalValidation.psm1') -Force

$repositoryRoot = Get-RepositoryRoot
$evidence = New-LocalEvidenceDirectory -EvidenceDirectory $EvidenceDirectory

[ordered]@{
    schema_version = '1.0.0'
    mode = 'validation_only'
    mutating_pipeline_stages = $false
    training = $false
    one_time_cohort_access = $false
    export_or_publication = $false
    note = 'Data processing, training, cohort evaluation, and publication require separate reviewed commands.'
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $evidence 'execution-scope.json') -Encoding UTF8

try {
    & (Join-Path $PSScriptRoot 'run_ml_validation.ps1') -EvidenceDirectory $evidence
    if ($LASTEXITCODE -ne 0) { throw 'ML validation runner failed' }
    & (Join-Path $PSScriptRoot 'run_android_validation.ps1') -EvidenceDirectory $evidence
    if ($LASTEXITCODE -ne 0) { throw 'Android validation runner failed' }
    & (Join-Path $PSScriptRoot 'verify_release_gate.ps1') -EvidenceDirectory $evidence
    Write-LocalRunManifest -EvidenceDirectory $evidence -Runner 'run_full_pipeline' -Status 'passed'
} catch {
    Write-LocalRunManifest -EvidenceDirectory $evidence -Runner 'run_full_pipeline' -Status 'failed_or_blocked'
    throw
}
