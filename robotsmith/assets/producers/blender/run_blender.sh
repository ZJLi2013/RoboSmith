#!/usr/bin/env bash
# Run a Blender Python script headless from this directory: ./run_blender.sh <script.py>
exec "$HOME/tools/blender" -b --factory-startup --python "$1" 2>&1 | grep -vE "^(Blender|Read prefs|Fra:|Saved:|Time:|Color management)"
