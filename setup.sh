#!/bin/bash
python3 -m venv .
source bin/activate
python -m pip install -r requirements.txt
direnv allow 2>&1 > /dev/null || true
