# Sørg for full UTF-8 støtte i PowerShell og Windows-konsoll
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 >$null

Set-Location -Path $PSScriptRoot
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"

Write-Host "========================================================================" -ForegroundColor Magenta
Write-Host " Starter Trigonometrisk Ekstremanalyse & Rekonstruksjon (Rust UI)..." -ForegroundColor White
Write-Host " (MERK: KUN sinus- og cosinusfunksjoner støttes)" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Magenta

if (Test-Path "target\debug\gpu_extrema_classifier.exe") {
    & ".\target\debug\gpu_extrema_classifier.exe"
} else {
    cargo run
}
