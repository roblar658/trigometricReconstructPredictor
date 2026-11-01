# ==============================================================================
# Interaktiv Terminal-Klient for Sanntids Sensor-Service (TCP)
# ==============================================================================
# Dette skriptet kobler til den kjørende sanntidstjenesten og lar deg sende:
#  - "1", "2", "3"  -> Sender automatisk 1, 2 eller 3 sensorverdier
#  - "x, y"         -> Sender et egendefinert datapunkt (f.eks. "0.5, 1.2")
#  - "batch"        -> Sender en testbatch på 10 punkter
#  - "status"       -> Viser nåværende tilpasset funksjon f(x) og antall punkter N
#  - "reset"        -> Nullstiller modellen
#  - "quit"         -> Avslutter tilkoblingen rent
# ==============================================================================

param (
    [int]$Port = 8765,
    [string]$HostAddr = "127.0.0.1"
)

$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

# Finn eksekverbar fil
$exePaths = @(
    (Join-Path $PSScriptRoot "target\release\gpu_extrema_classifier.exe"),
    (Join-Path $PSScriptRoot "gpu_extrema_classifier.exe"),
    "gpu_extrema_classifier.exe"
)

$exe = $null
foreach ($p in $exePaths) {
    if (Test-Path $p) {
        $exe = $p
        break
    }
}

if (-not $exe) {
    Write-Host "Bygger prosjektet først..." -ForegroundColor Yellow
    cargo build --release -j 1
    $exe = Join-Path $PSScriptRoot "target\release\gpu_extrema_classifier.exe"
}

Write-Host "Kobler til sensor-service på $HostAddr`:$Port..." -ForegroundColor Cyan
& $exe client --port $Port --host $HostAddr

