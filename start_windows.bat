@echo off
cd /d "%~dp0"
if not exist ".venv\installiert.ok" (
    echo Sicherheitscheck wird beim ersten Start eingerichtet. Das dauert einen Moment ...
    if not exist ".venv" (
        py -3 -m venv .venv 2>nul || python -m venv .venv
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (
        echo.
        echo Die Installation ist fehlgeschlagen. Bitte pruefe die Internetverbindung und die Python-Version ^(3.10 bis 3.12 empfohlen^).
        pause
        exit /b 1
    )
    echo ok> ".venv\installiert.ok"
)
if not exist ".venv\einrichtung2.ok" (
    ".venv\Scripts\python.exe" verknuepfung.py && echo ok> ".venv\einrichtung2.ok"
)
start "" ".venv\Scripts\pythonw.exe" sicherheitscheck.py %*
