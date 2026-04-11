@echo off
setlocal
cd /d "%~dp0"

set "OPENAI_LIVE_API_ALLOWED=1"
set "START_NODE=session_phase4_post_countdown_entry"
set "SEMANTIC_PROVIDER=openai-router"
set "APPROVAL_FILE=%~dp0live_api_approval.json"
set "MAX_API_CALLS=40"
set "SESSION_SANDBOX_TTS_PROVIDER=google"
set "SESSION_SANDBOX_TTS_VOICE_NAME=de-DE-Chirp3-HD-Iapetus"
set "SESSION_SANDBOX_TTS_LEAD_IN_MS=220"
set "SESSION_SANDBOX_TTS_POST_BLOCK_PAUSE_MS=380"
set "SPEAK_FLAG=--speak"
set "PYTHON_EXE=python"

if exist ".venv-router312\Scripts\python.exe" (
  set "PYTHON_EXE=.venv-router312\Scripts\python.exe"
) else (
  if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
  )
)

if not exist "%APPROVAL_FILE%" (
  echo [FEHLER] Freigabedatei fehlt: "%APPROVAL_FILE%"
  echo Lege zuerst backend\live_api_approval.json an oder pruefe den Pfad.
  exit /b 1
)

echo [LIVE-TEST] Phase-4 Full Flow ab Magic Chair
echo Startknoten: %START_NODE%
echo Semantik: %SEMANTIC_PROVIDER%
echo Max API Calls: %MAX_API_CALLS%
echo Approval File: %APPROVAL_FILE%
echo Python: %PYTHON_EXE%
echo.

"%PYTHON_EXE%" run_session_sandbox.py ^
  --node %START_NODE% ^
  --semantic-provider %SEMANTIC_PROVIDER% ^
  --live-api ^
  --max-api-calls %MAX_API_CALLS% ^
  --approval-file "%APPROVAL_FILE%" ^
  %SPEAK_FLAG%

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo [FEHLER] Live-Test wurde mit Exit-Code %EXIT_CODE% beendet.
)
exit /b %EXIT_CODE%
