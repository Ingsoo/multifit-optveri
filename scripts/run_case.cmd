@echo off
if "%~2"=="" (
  echo Usage: scripts\run_case.cmd MACHINE JOB
  exit /b 1
)
conda run --live-stream -n multifit-optveri python scripts\run_obv.py run --config configs\experiments\paper_base.toml --machine %1 --job %2
