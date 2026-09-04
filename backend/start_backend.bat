@echo off
REM Always use system Python 3.11 (has TensorFlow). Do not use backend\venv.
cd /d "%~dp0"
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2
if not exist .env copy .env.example .env >nul

powershell -NoProfile -Command "try { $r = Invoke-WebRequest 'http://127.0.0.1:5000/api/health' -UseBasicParsing -TimeoutSec 2; if ($r.Content -match '\"model_loaded\":\s*true') { Write-Host 'Backend already running with model loaded at http://127.0.0.1:5000'; exit 0 }; Write-Host 'Port 5000 is in use but the model is not loaded. Close the old python/app.py terminal and run this again.'; exit 1 } catch { exit 2 }"
if %ERRORLEVEL%==0 exit /b 0
if %ERRORLEVEL%==1 exit /b 1

set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if exist "%PY%" goto run
where py >nul 2>&1 && (echo Starting PneumoDetect backend on http://localhost:5000 ... & py -3.11 app.py & exit /b %ERRORLEVEL%)
where python >nul 2>&1 && (echo WARNING: Python 3.11 not found. Using default python — TensorFlow may fail. & python app.py & exit /b %ERRORLEVEL%)
echo Python 3.11 with TensorFlow was not found.
exit /b 1

:run
echo Starting PneumoDetect backend with "%PY%" on http://localhost:5000 ...
"%PY%" app.py
