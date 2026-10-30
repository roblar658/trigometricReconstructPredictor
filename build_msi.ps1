# build_msi.ps1 - Bygg MSI-installasjonspakke med WiX Toolset
param (
    [switch]$DebugBuild
)

[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 >$null

Set-Location -Path $PSScriptRoot

$exePath = if ($DebugBuild) { "target\debug\gpu_extrema_classifier.exe" } else { "target\release\gpu_extrema_classifier.exe" }

if (-not (Test-Path $exePath)) {
    Write-Host "Kompilerer programvare ($exePath)..." -ForegroundColor Yellow
    if ($DebugBuild) {
        cargo build
    } else {
        cargo build --release -j 1
    }
}

if (-not (Test-Path "target\wix")) {
    New-Item -ItemType Directory -Path "target\wix" -Force | Out-Null
}

$msiOut = "target\wix\TrigonometriskEkstremanalyse-Setup-v1.1.0.msi"

Write-Host "Bygger MSI-installasjonspakke med WiX Toolset v4..." -ForegroundColor Cyan
wix build wix\main.wxs -ext WixToolset.UI.wixext -o $msiOut

if ($LASTEXITCODE -eq 0) {
    Write-Host "✔ MSI-installasjonspakke bygget med suksess!" -ForegroundColor Green
    Write-Host "  Filsti: $msiOut" -ForegroundColor White
} else {
    Write-Host "❌ Feil under bygging av MSI (feilkode $LASTEXITCODE)" -ForegroundColor Red
}

