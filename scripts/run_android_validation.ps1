param([string]$EvidenceDirectory)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'local\LocalValidation.psm1') -Force

$repositoryRoot = Get-RepositoryRoot
$androidRoot = Join-Path $repositoryRoot 'android'
$evidence = New-LocalEvidenceDirectory -EvidenceDirectory $EvidenceDirectory
$internalModelIndex = Join-Path $repositoryRoot 'data\artifacts\runs\20260916T132736Z\export\internal\internal_model_index.json'

if (-not (Test-Path -LiteralPath $internalModelIndex -PathType Leaf)) {
    throw "Production internal model index is required for Android validation: $internalModelIndex"
}

try {
    Invoke-LocalGateCommand -Name '07_android_host' -Executable '.\gradlew.bat' `
        -Arguments @(
            '--no-daemon', "--project-prop=internalModelIndex=$internalModelIndex",
            ':shared:test', ':inference:testProductionDebugUnitTest', ':app:testProductionDebugUnitTest', ':wear:testProductionDebugUnitTest',
            ':app:assembleProductionDebug', ':wear:assembleProductionDebug'
        ) -EvidenceDirectory $evidence -WorkingDirectory $androidRoot
    Write-LocalRunManifest -EvidenceDirectory $evidence -Runner 'run_android_validation' -Status 'passed'
    Write-Host "Local Android validation passed. Evidence: $evidence"
} catch {
    Write-LocalRunManifest -EvidenceDirectory $evidence -Runner 'run_android_validation' -Status 'failed'
    throw
}
