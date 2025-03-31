@echo off
REM Shortcut batch file for vAIn system commands

if "%1"=="" goto help
if "%1"=="--help" goto help
if "%1"=="/?" goto help
if "%1"=="help" goto help
if "%1"=="config" goto config_help

REM Pass all arguments to Python script
python %~dp0\tools\terminal_commands.py %*
goto :eof

:help
echo.
echo vAIn System Command Utility
echo --------------------------
echo Usage: va [command] [arguments]
echo.
echo Common Commands:
echo   l0              Start in standard mode
echo   l1              Start in debug mode
echo   c0              List available configurations
echo   c1 [config]     Show specific configuration
echo   c2 [config]     Update specific configuration
echo   c3              Update all configurations
echo   help            Show this help text
echo   config          Show configuration help
echo.
python %~dp0\tools\terminal_commands.py help
goto :eof

:config_help
echo.
echo vAIn Configuration Management
echo --------------------------
echo Usage: va config [command] [arguments]
echo.
echo Configuration Commands:
echo   va c0                           List all available configurations
echo   va c1 network                   Show network configuration
echo   va c2 network                   Update network configuration
echo   va c3 --level=normal            Update all configs with normal interaction level
echo   va c3 --level=minimal           Update essential configs with minimal interaction
echo   va c3 --level=verbose           Update all configs with verbose interaction
echo   va c4 blockchain                Validate blockchain configuration
echo.
echo Direct Command Access:
echo   va config list                  List all available configurations
echo   va config show [name]           Show specific configuration
echo   va config update [name]         Update specific configuration
echo   va config validate [name]       Validate specific configuration
echo   va config update-all            Update all configurations
echo   va config create                Create default configurations
echo.

REM If command arguments were provided (config + something else), run that command
if not "%2"=="" (
    python %~dp0\tools\config_manager.py %2 %3 %4 %5 %6 %7 %8 %9
)
goto :eof
