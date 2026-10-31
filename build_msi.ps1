# build_msi.ps1 - Bygg MSI-installasjonspakke med WiX Toolset
param (
    [switch]$DebugBuild,
    [switch]$SkipCompile
)

[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 >$null

Set-Location -Path $PSScriptRoot

$exePath = if ($DebugBuild) { "target\debug\gpu_extrema_classifier.exe" } else { "target\release\gpu_extrema_classifier.exe" }

# Kompiler alltid oppdatert binærfil med mindre -SkipCompile er angitt
if (-not $SkipCompile) {
    Write-Host "Kompilerer oppdatert programvare ($exePath)..." -ForegroundColor Yellow
    if ($DebugBuild) {
        cargo build -j 1
    } else {
        cargo build --release -j 1
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FEIL] Kompilering feilet." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

if (-not (Test-Path $exePath)) {
    Write-Host "[FEIL] Fant ikke $exePath for pakking i MSI." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "target\wix")) {
    New-Item -ItemType Directory -Path "target\wix" -Force | Out-Null
}

$msiOut = "target\wix\TrigonometriskEkstremanalyse-Setup-v1.1.0.msi"

Write-Host "Bygger MSI-installasjonspakke med WiX Toolset v4 ($exePath)..." -ForegroundColor Cyan
wix build wix\main.wxs -ext WixToolset.UI.wixext -d ExeSource=$exePath -o $msiOut

if ($LASTEXITCODE -eq 0) {
    if (-not (Test-Path "install")) {
        New-Item -ItemType Directory -Path "install" -Force | Out-Null
    }
    $installMsi = "install\TrigonometriskEkstremanalyse-Setup-v1.1.0.msi"
    try {
        Copy-Item $msiOut $installMsi -Force -ErrorAction Stop
        Write-Host "  Filsti (install): $installMsi" -ForegroundColor Cyan
    } catch {
        Write-Host "  [OBS] Kunne ikke overskrive $installMsi (filen er trolig åpen i Windows Installer/msiexec)." -ForegroundColor Yellow
        Write-Host "  Bruk den oppdaterte filen direkte fra: $msiOut" -ForegroundColor Yellow
    }

    Write-Host "[OK] MSI-installasjonspakke bygget med suksess!" -ForegroundColor Green
    Write-Host "  Innebygd binærfil: $exePath" -ForegroundColor White
    Write-Host "  Filsti (bygg):    $msiOut" -ForegroundColor White
    Write-Host "  Filsti (install): $installMsi" -ForegroundColor Cyan
} else {
    Write-Host "[FEIL] Feil under bygging av MSI (feilkode $LASTEXITCODE)" -ForegroundColor Red
}
