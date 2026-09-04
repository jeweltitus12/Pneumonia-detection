@echo off
cd /d "%~dp0"
echo Starting PneumoDetect backend and frontend...
echo Frontend: http://127.0.0.1:5173
echo Backend:  http://127.0.0.1:5000
start "PneumoDetect Backend" cmd /k "%~dp0backend\start_backend.bat"
start "PneumoDetect Frontend" cmd /k "%~dp0frontend\start_frontend.bat"
echo Wait about 30-90 seconds for "Model loaded" in the backend window, then refresh the app.
