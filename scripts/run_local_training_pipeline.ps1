param(
    [switch]$Resume,
    [switch]$Smoke,
    [ValidatePattern('^\d{8}T\d{6}Z$')]
    [string]$RunId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$arguments = @(
    'all',
    '--config', 'PipelinePython/configs/data.yaml',
    '--run-id', $RunId
)
if ($Resume) { $arguments += '--resume' }
if ($Smoke) { $arguments += '--smoke' }

# Canonical stages include data preparation, peak validation, training, and
# gated ONNX export.
# Their durable state is checksum-bound in the run's run_state.json.
Push-Location $repositoryRoot
try {
    & rtk python -m PipelinePython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Canonical local pipeline failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
