#!/usr/bin/env bash
set -eu
cd /tmp/robotsmith-e2e
export PYTHONPATH=/tmp/robotsmith-e2e
python3 scripts/test_e2e_remote.py
