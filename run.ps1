# run.ps1 - Start Trigonometrisk Ekstremanalyse & Rekonstruksjon
param (
    [switch]$DebugMode,
    [switch]$Build,
    [switch]$Diagram,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

# Sørg for full UTF-8 støtte i PowerShell og Windows-konsoll
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 >$null

Set-Location -Path $PSScriptRoot

if ($Diagram) {
    Write-Host "========================================================================" -ForegroundColor Magenta
    Write-Host " ⚡ SYSTEMARKITEKTUR, DATAFLYT OG TILSTANDSDIAGRAMMER" -ForegroundColor White
    Write-Host "========================================================================" -ForegroundColor Magenta
    Write-Host " 1. Brukergrensesnitt (egui/eframe) -> 4 Faner: Funksjon, CSV/FLAC, Sensor, Hjelp" -ForegroundColor Cyan
    Write-Host " 2. Lyddekoding & Spektralanalyse: claxon -> Trapesintegrasjon -> OMP -> LeastSquares" -ForegroundColor Cyan
    Write-Host " 3. GPU Beregningskjerne (wgpu / WGSL): 256 tråder -> Newton-Raphson rotfinner" -ForegroundColor Cyan
    Write-Host " 4. Klassifisering: Klasse 0-4, Terskler T1-T4, Topp- og bunnpunkter" -ForegroundColor Cyan
    Write-Host " 5. Sanntids Sensor-Service: O(1) Inkrementell Gram -> TCP & HTTP GET (JSON)" -ForegroundColor Cyan
    Write-Host " 6. Rapportering: Chromium Headless PDF-eksport & WiX MSI" -ForegroundColor Cyan
    Write-Host "========================================================================" -ForegroundColor Magenta

    $diagHtml = Join-Path $PSScriptRoot "diagrams.html"
    if (Test-Path $diagHtml) {
        Write-Host " Åpner fullstendige interaktive Mermaid-diagrammer i nettleseren..." -ForegroundColor Green
        Start-Process $diagHtml
    } else {
        Write-Host " Fant ikke $diagHtml. Se diagrammene i README.md" -ForegroundColor Yellow
    }
    exit 0
}

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
