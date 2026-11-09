#!/usr/bin/env bash
# Startet den Sicherheitscheck. Beim ersten Start wird eine eigene Python-Umgebung eingerichtet.
set -e
cd "$(dirname "$0")"

if [ ! -f ".venv/installiert.ok" ]; then
    echo "Sicherheitscheck wird beim ersten Start eingerichtet. Das dauert einen Moment …"
    [ -d ".venv" ] || python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    if ! .venv/bin/python -m pip install -r requirements.txt; then
        echo
        echo "Die Installation ist fehlgeschlagen. Bitte prüfe die Internetverbindung und die Python-Version (3.10 bis 3.12 empfohlen)."
        exit 1
    fi
    touch .venv/installiert.ok
fi

if [ ! -f ".venv/einrichtung2.ok" ]; then
    .venv/bin/python verknuepfung.py && touch .venv/einrichtung2.ok || true
fi

exec .venv/bin/python sicherheitscheck.py "$@"
