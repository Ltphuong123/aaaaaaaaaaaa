@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-voice-studio.ps1"
if errorlevel 1 pause
