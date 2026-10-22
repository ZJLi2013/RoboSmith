"""Build the corrected drawer cabinet asset.

The source is kept in previous_build.py so the rejected geometry and its fix
remain directly reviewable in one place.
"""

from pathlib import Path


source = Path(__file__).with_name("previous_build.py")
exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"))
