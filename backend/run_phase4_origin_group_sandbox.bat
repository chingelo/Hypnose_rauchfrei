@echo off
cd /d "%~dp0"
set "SESSION_SANDBOX_TTS_PROVIDER=google"
set "SESSION_SANDBOX_TTS_VOICE_NAME=de-DE-Chirp3-HD-Iapetus"
set "SESSION_SANDBOX_TTS_LEAD_IN_MS=220"
set "SESSION_SANDBOX_TTS_POST_BLOCK_PAUSE_MS=380"
if exist ".venv\Scripts\python.exe" (
  .venv\Scripts\python.exe run_session_sandbox.py --node dark_origin_terminal --speak
) else (
  python run_session_sandbox.py --node dark_origin_terminal --speak
)
