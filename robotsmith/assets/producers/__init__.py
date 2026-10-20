"""Asset producers: 3D generation backends and mesh -> sim-ready URDF conversion.

Producers write an asset directory (``model.urdf``, meshes, ``generation.json``);
``AssetLibrary`` loads it. Generation backends import their model repos lazily
and are meant to run in a separate image from the Genesis stack.
"""

from robotsmith.assets.producers.backend import (
    BackendInfo,
    GenBackend,
    get_backend,
    list_backend_info,
    list_backends,
    register_backend,
)
from robotsmith.assets.producers.mesh_to_urdf import mesh_to_urdf
from robotsmith.assets.producers.produce import generate_asset

__all__ = [
    "BackendInfo",
    "GenBackend",
    "get_backend",
    "list_backend_info",
    "list_backends",
    "register_backend",
    "mesh_to_urdf",
    "generate_asset",
]
