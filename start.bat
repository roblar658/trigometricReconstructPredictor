@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
title Trigonometrisk Ekstremanalyse ^& Rekonstruksjon

echo ========================================================================
echo  Starter Trigonometrisk Ekstremanalyse ^& Rekonstruksjon (Rust UI)...
echo  (MERK: KUN sinus- og cosinusfunksjoner støttes)
echo ========================================================================

if exist "target\debug\gpu_extrema_classifier.exe" (
    "target\debug\gpu_extrema_classifier.exe"
) else (
    cargo run
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo Programmet avsluttet med feilkode %ERRORLEVEL%.
    pause
)
