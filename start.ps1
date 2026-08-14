$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ToolsDir = Join-Path $RootDir ".surveyflow-tools"
$LocalUv = Join-Path $ToolsDir "uv.exe"
$UvCommand = Get-Command uv -ErrorAction SilentlyContinue

if ($UvCommand) {
    $Uv = $UvCommand.Source
} elseif (Test-Path $LocalUv) {
    $Uv = $LocalUv
} else {
    New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
    Write-Host "uv was not found. Installing a private copy for SurveyFlow..."
    $env:UV_INSTALL_DIR = $ToolsDir
    $env:UV_NO_MODIFY_PATH = "1"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $Uv = $LocalUv
}

Set-Location $RootDir
Write-Host "Preparing SurveyFlow. The first start may download Python and dependencies..."
& $Uv sync --frozen --no-dev
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Port = if ($env:SURVEYFLOW_PORT) { $env:SURVEYFLOW_PORT } else { "8501" }
Write-Host "Opening SurveyFlow at http://localhost:$Port"
& $Uv run vnn-survey-app
exit $LASTEXITCODE
