#!/bin/bash -e
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
direnv allow 2>&1 > /dev/null || true
