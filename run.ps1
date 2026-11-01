# run.ps1 - Start Trigonometrisk Ekstremanalyse & Rekonstruksjon
param (
    [switch]$DebugMode,
    [switch]$Build,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

# Sørg for full UTF-8 støtte i PowerShell og Windows-konsoll
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 >$null

Set-Location -Path $PSScriptRoot
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"

$exePath = if ($DebugMode) {
    "target\debug\gpu_extrema_classifier.exe"
} else {
    "target\release\gpu_extrema_classifier.exe"
}

# Kompiler dersom -Build er angitt, eller hvis binærfilen ikke eksisterer
if ($Build -or (-not (Test-Path $exePath))) {
    if ($DebugMode) {
        Write-Host "Kompilerer debug-versjon (cargo build -j 1)..." -ForegroundColor Yellow
        cargo build -j 1
    } else {
        Write-Host "Kompilerer optimalisert release-versjon (cargo build --release -j 1)..." -ForegroundColor Yellow
        cargo build --release -j 1
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FEIL] Kompilering feilet." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# Fallback hvis foretrukket versjon ikke finnes
if (-not (Test-Path $exePath)) {
    if (Test-Path "target\release\gpu_extrema_classifier.exe") {
        $exePath = "target\release\gpu_extrema_classifier.exe"
    } elseif (Test-Path "target\debug\gpu_extrema_classifier.exe") {
        $exePath = "target\debug\gpu_extrema_classifier.exe"
    }
}

Write-Host "========================================================================" -ForegroundColor Magenta
Write-Host " Starter Trigonometrisk Ekstremanalyse & Rekonstruksjon..." -ForegroundColor White
Write-Host " Kjører: $exePath" -ForegroundColor Gray
Write-Host " (MERK: KUN sinus- og cosinusfunksjoner støttes)" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Magenta

if (Test-Path $exePath) {
    if ($AppArgs -and $AppArgs.Count -gt 0) {
        & ".\$exePath" @AppArgs
    } else {
        & ".\$exePath"
    }
} else {
    Write-Host "[FEIL] Fant ikke kjørbar fil. Prøv å kjøre med -Build flagget: .\run.ps1 -Build" -ForegroundColor Red
}
