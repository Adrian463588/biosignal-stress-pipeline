Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-RepositoryRoot {
    return [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
}

function Get-LocalSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolved = [System.IO.Path]::GetFullPath($Path)
    if (-not [System.IO.File]::Exists($resolved)) {
        throw "Cannot hash missing file: $resolved"
    }
    $stream = [System.IO.File]::OpenRead($resolved)
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = $algorithm.ComputeHash($stream)
        return ([System.BitConverter]::ToString($bytes)).Replace('-', '').ToLowerInvariant()
    } finally {
        $algorithm.Dispose()
        $stream.Dispose()
    }
}

function New-LocalEvidenceDirectory {
    param([string]$EvidenceDirectory)

    $repositoryRoot = Get-RepositoryRoot
    $allowedRoot = [System.IO.Path]::GetFullPath(
        (Join-Path $repositoryRoot 'build\release-evidence-local\runs')
    )
    if ([string]::IsNullOrWhiteSpace($EvidenceDirectory)) {
        $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
        $resolved = Join-Path $allowedRoot $stamp
    } else {
        $resolved = [System.IO.Path]::GetFullPath($EvidenceDirectory)
    }

    $prefix = $allowedRoot.TrimEnd('\') + '\'
    if (-not ($resolved + '\').StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Evidence directory must remain under $allowedRoot; found $resolved"
    }
    New-Item -ItemType Directory -Path $resolved -Force | Out-Null
    return $resolved
}

function Invoke-LocalGateCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$EvidenceDirectory,
        [string]$WorkingDirectory = (Get-RepositoryRoot)
    )

    $safeName = $Name -replace '[^A-Za-z0-9_.-]', '_'
    $logPath = Join-Path $EvidenceDirectory "$safeName.log"
    $recordPath = Join-Path $EvidenceDirectory "$safeName.json"
    $started = [DateTime]::UtcNow
    $previousLocation = Get-Location
    $exitCode = 1
    try {
        Set-Location -LiteralPath $WorkingDirectory
        & $Executable @Arguments 2>&1 | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE
        if ($null -eq $exitCode) {
            $exitCode = 0
        }
    } finally {
        Set-Location -LiteralPath $previousLocation
        $finished = [DateTime]::UtcNow
        [ordered]@{
            schema_version = '1.0.0'
            name = $Name
            executable = $Executable
            arguments = $Arguments
            working_directory = $WorkingDirectory
            started_at_utc = $started.ToString('o')
            finished_at_utc = $finished.ToString('o')
            duration_seconds = [Math]::Round(($finished - $started).TotalSeconds, 3)
            exit_code = $exitCode
            log_file = [System.IO.Path]::GetFileName($logPath)
        } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $recordPath -Encoding UTF8
    }
    if ($exitCode -ne 0) {
        throw "Local gate '$Name' failed with exit code $exitCode. See $logPath"
    }
}

function Write-LocalRunManifest {
    param(
        [Parameter(Mandatory = $true)][string]$EvidenceDirectory,
        [Parameter(Mandatory = $true)][string]$Runner,
        [Parameter(Mandatory = $true)][string]$Status
    )

    $repositoryRoot = Get-RepositoryRoot
    $revision = (& git -C $repositoryRoot rev-parse HEAD 2>$null)
    $gateConfig = Join-Path $repositoryRoot 'PipelinePython\configs\frozen_gates.yaml'
    [ordered]@{
        schema_version = '1.0.0'
        runner = $Runner
        status = $Status
        generated_at_utc = [DateTime]::UtcNow.ToString('o')
        repository_root = $repositoryRoot
        git_revision = if ($LASTEXITCODE -eq 0) { $revision } else { $null }
        gate_config_sha256 = Get-LocalSha256 -Path $gateConfig
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (
        Join-Path $EvidenceDirectory "$Runner.manifest.json"
    ) -Encoding UTF8
}

Export-ModuleMember -Function Get-RepositoryRoot, New-LocalEvidenceDirectory, Invoke-LocalGateCommand, Write-LocalRunManifest
