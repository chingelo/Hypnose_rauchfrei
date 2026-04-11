@echo off
cd /d "%~dp0"
set "OPENAI_LIVE_API_ALLOWED=1"
set "APPROVAL_FILE=C:\Projekte\test_app\backend\live_api_approval.json"
set "MAX_API_CALLS=40"
set "SESSION_SANDBOX_TTS_PROVIDER=google"
set "SESSION_SANDBOX_TTS_VOICE_NAME=de-DE-Chirp3-HD-Iapetus"
set "SESSION_SANDBOX_TTS_LEAD_IN_MS=220"
set "SESSION_SANDBOX_TTS_POST_BLOCK_PAUSE_MS=380"
if exist ".venv-router312\Scripts\python.exe" (
  .venv-router312\Scripts\python.exe run_session_sandbox.py --node session_phase4_intro --semantic-provider openai-router --live-api --max-api-calls %MAX_API_CALLS% --approval-file "%APPROVAL_FILE%" --speak
) else (
  if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe run_session_sandbox.py --node session_phase4_intro --semantic-provider openai-router --live-api --max-api-calls %MAX_API_CALLS% --approval-file "%APPROVAL_FILE%" --speak
  ) else (
    python run_session_sandbox.py --node session_phase4_intro --semantic-provider openai-router --live-api --max-api-calls %MAX_API_CALLS% --approval-file "%APPROVAL_FILE%" --speak
  )
)
