@echo off
cd /d "C:\Projekte\test_app\backend"
set HF_HUB_DISABLE_XET=1
set HF_HUB_DISABLE_PROGRESS_BARS=1
if exist ".venv-router312\Scripts\python.exe" (
  .venv-router312\Scripts\python.exe train_local_router_model.py --base-model Qwen/Qwen2.5-3B-Instruct --output-dir C:\Projekte\test_app\backend\finetune_data\v2\local_router\artifacts\router_qlora_qwen25_3b --dry-run
) else (
  python train_local_router_model.py --base-model Qwen/Qwen2.5-3B-Instruct --output-dir C:\Projekte\test_app\backend\finetune_data\v2\local_router\artifacts\router_qlora_qwen25_3b --dry-run
)
pause
