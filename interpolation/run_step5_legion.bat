@echo off
REM Step 5 on the Legion (GPU): InterpolAI test on every ROI, structure metrics, pooled statistics.
REM Run over SSH:   D:\thesis\InterpolAI\run_step5_legion.bat
REM PY = python.exe of the InterpolAI conda env (check with: conda env list)
set PY=D:\miniconda3\envs\InterpolAI\python.exe
set DATA=D:\thesis\data
cd /d D:\thesis\InterpolAI

for %%R in (roi1 roi2 roi3) do (
  echo ===== %%R =====
  "%PY%" liver_step5_interpolai_eval.py --pth %DATA%\%%R --skip 1 3 > %DATA%\%%R_step5.log 2>&1
  "%PY%" liver_step5b_structure_metrics.py --pth %DATA%\%%R --skip 1 3
)
"%PY%" liver_step5c_pool_rois.py %DATA%\roi1 %DATA%\roi2 %DATA%\roi3
