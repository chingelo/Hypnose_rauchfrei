@echo off
setlocal
cd /d C:\Projekte\test_app\backend
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe run_phase4_semantic_ft_prototype.py --node hell_feel_branch
) else (
  python run_phase4_semantic_ft_prototype.py --node hell_feel_branch
)
endlocal
