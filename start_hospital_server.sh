#!/usr/bin/env bash
# Start Micro-QA + TRS-398 Hospital Server on Local Network
cd "$(dirname "$0")"

# Activate venv if present
if [ -d "venv" ]; then
    ./venv/bin/python3 hospital_server.py "$@"
else
    python3 hospital_server.py "$@"
fi
