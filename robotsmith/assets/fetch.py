"""``python -m robotsmith.assets.fetch``: download the build files this checkout's ``asset.json`` files pin."""

import sys

from robotsmith.assets.package import main

main(["fetch", *sys.argv[1:]])
