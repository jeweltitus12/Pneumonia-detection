@echo off
cd /d "%~dp0"
echo Starting PneumoDetect frontend on http://127.0.0.1:5173 ...
npm run dev -- --port 5173 --host 127.0.0.1
