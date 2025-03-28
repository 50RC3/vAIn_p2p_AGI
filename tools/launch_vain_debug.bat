@echo off
REM Launch vAIn debugging session with port conflict handling
cd %~dp0\..
python tools\debug_launcher.py main.py --auto-resolve
