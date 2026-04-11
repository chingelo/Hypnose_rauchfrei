@echo off
cd /d "%~dp0"
set "SESSION_SANDBOX_TTS_PROVIDER=google"
set "SESSION_SANDBOX_TTS_VOICE_NAME=de-DE-Chirp3-HD-Iapetus"
set "SESSION_SANDBOX_TTS_LEAD_IN_MS=220"
set "SESSION_SANDBOX_TTS_POST_BLOCK_PAUSE_MS=380"
if exist ".venv-router312\Scripts\python.exe" (
  .venv-router312\Scripts\python.exe run_session_sandbox.py --node session_phase4_intro --semantic-provider local-intent --local-intent-adapter-dir C:\Projekte\test_app\backend\finetune_data\v3\local_router_intent\artifacts\router_qlora_qwen25_3b_v3_intent --speak
) else (
  if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe run_session_sandbox.py --node session_phase4_intro --semantic-provider local-intent --local-intent-adapter-dir C:\Projekte\test_app\backend\finetune_data\v3\local_router_intent\artifacts\router_qlora_qwen25_3b_v3_intent --speak
  ) else (
    python run_session_sandbox.py --node session_phase4_intro --semantic-provider local-intent --local-intent-adapter-dir C:\Projekte\test_app\backend\finetune_data\v3\local_router_intent\artifacts\router_qlora_qwen25_3b_v3_intent --speak
  )
)
