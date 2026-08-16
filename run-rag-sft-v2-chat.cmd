@echo off
setlocal
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run-rag-sft-v2-chat.ps1"
set "chat_exit_code=%ERRORLEVEL%"
if not "%chat_exit_code%"=="0" (
    echo.
    echo Startup failed. See the error message above.
    pause
)
exit /b %chat_exit_code%
