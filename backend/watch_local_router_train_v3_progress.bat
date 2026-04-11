@echo off
cd /d "C:\Projekte\test_app\backend"
powershell -NoExit -Command ^
  "$stdout='C:\Projekte\test_app\backend\finetune_data\v3\local_router\artifacts\logs\train_qwen3b_v3_stdout.log';" ^
  "$stderr='C:\Projekte\test_app\backend\finetune_data\v3\local_router\artifacts\logs\train_qwen3b_v3_stderr.log';" ^
  "Write-Host 'STDOUT zuletzt:' -ForegroundColor Cyan;" ^
  "if (Test-Path $stdout) { Get-Content $stdout -Tail 25 } else { Write-Host 'stdout log noch nicht vorhanden.' -ForegroundColor Yellow };" ^
  "Write-Host '';" ^
  "Write-Host 'Live-Step-Fortschritt aus STDERR:' -ForegroundColor Green;" ^
  "if (Test-Path $stderr) { Get-Content $stderr -Wait -Tail 80 } else { Write-Host 'stderr log noch nicht vorhanden.' -ForegroundColor Yellow }"
