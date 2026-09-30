param(
    [string]$EvidenceDirectory,
    [switch]$EvidenceOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'local\LocalValidation.psm1') -Force

$evidence = New-LocalEvidenceDirectory -EvidenceDirectory $EvidenceDirectory
$snapshotDirectory = Join-Path $evidence 'release-snapshot'

Invoke-LocalGateCommand -Name '08_release_snapshot' -Executable 'rtk' `
    -Arguments @(
        'python', 'PipelinePython/03_stress_onnx/tools/generate_release_evidence.py',
        '--output-dir', $snapshotDirectory
    ) -EvidenceDirectory $evidence

$snapshot = Get-Content -Raw -LiteralPath (
    Join-Path $snapshotDirectory 'release_status.json'
) | ConvertFrom-Json

if ($snapshot.release_status -ne 'candidate') {
    Write-LocalRunManifest -EvidenceDirectory $evidence -Runner 'verify_release_gate' -Status 'blocked'
    if ($EvidenceOnly) {
        Write-Host "Release remains blocked; evidence snapshot created at $snapshotDirectory"
        return
    }
    throw "Release gate blocked with status '$($snapshot.release_status)'. See $snapshotDirectory"
}

Write-LocalRunManifest -EvidenceDirectory $evidence -Runner 'verify_release_gate' -Status 'passed'
Write-Host "Local release gate passed. Manual sign-off may now review: $snapshotDirectory"
