param([string]$EvidenceDirectory)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'local\LocalValidation.psm1') -Force

$repositoryRoot = Get-RepositoryRoot
$evidence = New-LocalEvidenceDirectory -EvidenceDirectory $EvidenceDirectory

try {
    Invoke-LocalGateCommand -Name '01_exact_gate' -Executable 'rtk' `
        -Arguments @(
            'python', 'PipelinePython/03_stress_onnx/tools/check_frozen_gate.py', '--base-ref', 'HEAD',
            '--adr-reference', 'ADR-013'
        ) -EvidenceDirectory $evidence
    Invoke-LocalGateCommand -Name '02_production_fake_scan' -Executable 'rtk' `
        -Arguments @('python', 'scripts/local/scan_android_production_fakes.py') -EvidenceDirectory $evidence
    Invoke-LocalGateCommand -Name '03_ruff' -Executable 'rtk' `
        -Arguments @('ruff', 'check', 'PipelinePython', 'scripts/local') -EvidenceDirectory $evidence
    Invoke-LocalGateCommand -Name '04_mypy_critical' -Executable 'rtk' `
        -Arguments @(
            'python', '-m', 'mypy', '--ignore-missing-imports', '--follow-imports=skip',
            'PipelinePython/common/config.py', 'PipelinePython/common/export_onnx_dl.py',
            'PipelinePython/common/cv/leakage.py', 'PipelinePython/common/cv/fold_validator.py'
        ) -EvidenceDirectory $evidence
    Invoke-LocalGateCommand -Name '05_pytest_full' -Executable 'rtk' `
        -Arguments @('python', '-m', 'pytest', '-q') -EvidenceDirectory $evidence
    Invoke-LocalGateCommand -Name '06_model_index' -Executable 'rtk' `
        -Arguments @('python', 'PipelinePython/03_stress_onnx/tools/validate_model_index.py') -EvidenceDirectory $evidence
    Write-LocalRunManifest -EvidenceDirectory $evidence -Runner 'run_ml_validation' -Status 'passed'
    Write-Host "Local ML validation passed. Evidence: $evidence"
} catch {
    Write-LocalRunManifest -EvidenceDirectory $evidence -Runner 'run_ml_validation' -Status 'failed'
    throw
}
