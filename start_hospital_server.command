#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ -d "venv" ]; then
    ./venv/bin/python3 hospital_server.py --browser
else
    python3 hospital_server.py --browser
fi
