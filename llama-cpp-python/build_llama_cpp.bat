@echo off
REM ==============================================
REM Llama.cpp Build Script - Cathedral Edition
REM ==============================================

SETLOCAL ENABLEDELAYEDEXPANSION

REM -- CONFIGURATION -----------------------------
SET "VSCOMPILER=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64\cl.exe"
SET "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
SET "LLAMA_PATH=%~dp0"

REM -- ENV SETUP ---------------------------------
echo [*] Setting up build environment...

SET "CMAKE_GENERATOR_PLATFORM=x64"
SET "CMAKE_CUDA_HOST_COMPILER=%VSCOMPILER%"
SET "CMAKE_ARGS=-DLLAMA_MAX_CONTEXT=2048 -DGGML_CUDA=on"
SET "LLAMA_CUDA_ARCHS=native"

REM -- PYTHON ENVIRONMENT ------------------------
echo [*] Activating virtual environment...
CALL ..\venv\Scripts\activate.bat

REM -- CLEAN BUILD -------------------------------
echo [*] Cleaning previous builds...
rd /s /q build 2>nul
rd /s /q dist 2>nul
del /f /q *.egg-info 2>nul

REM -- START BUILD -------------------------------
echo [*] Starting build...
pip install . --no-build-isolation --force-reinstall --no-cache-dir

IF %ERRORLEVEL% NEQ 0 (
    echo [!] Build failed.
    EXIT /B 1
) ELSE (
    echo [✓] Build completed successfully!
)

ENDLOCAL
