"""Abstract base class and registry for 3D generation backends.

Each backend wraps a specific image/text-to-3D model (Hunyuan3D, TRELLIS.2, TripoSG, etc.)
and returns a trimesh.Trimesh. The orchestrator (generate.py) handles downstream conversion
(mesh_to_urdf, catalog, library registration).

Usage::

    from robotsmith.gen.backend import get_backend, list_backends

    backend = get_backend("hunyuan3d")
    mesh = backend.generate("red ceramic mug", image_path="mug.png")

    print(list_backends())  # ['hunyuan3d', 'trellis2', 'triposg']
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import trimesh


@dataclass
class BackendInfo:
    """Metadata describing a generation backend's capabilities."""

    name: str
    model_name: str
    version: str = ""
    has_pbr: bool = False
    min_vram_gb: float = 0.0
    rocm_status: str = "unknown"  # "verified", "community_fork", "likely", "unknown"
    description: str = ""
    install_hint: str = ""


class GenBackend(ABC):
    """Abstract base class for 3D generation backends.

    Subclasses must implement `generate()` and `info`.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        output_path: Optional[str | Path] = None,
        **kwargs,
    ) -> trimesh.Trimesh:
        """Generate a 3D mesh from a text prompt.

        Args:
            prompt: Text description of the object to generate.
            output_path: Optional path to export the mesh file.
            **kwargs: Backend-specific parameters.

        Returns:
            A trimesh.Trimesh object.
        """
        ...

    @property
    @abstractmethod
    def info(self) -> BackendInfo:
        """Return metadata about this backend."""
        ...

    def is_available(self) -> bool:
        """Check whether this backend's dependencies are installed.

        Override in subclass for accurate checks.
        """
        return True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[GenBackend]] = {}


def register_backend(name: str):
    """Decorator to register a GenBackend subclass."""
    def decorator(cls: type[GenBackend]):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_backend(name: str, **init_kwargs) -> GenBackend:
    """Instantiate a registered backend by name.

    Raises KeyError if the name is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise KeyError(
            f"Unknown gen backend: {name!r}. Available: {available}"
        )
    return _REGISTRY[name](**init_kwargs)


def list_backends() -> list[str]:
    """Return names of all registered backends."""
    return sorted(_REGISTRY.keys())


def list_backend_info() -> list[BackendInfo]:
    """Return BackendInfo for all registered backends."""
    infos = []
    for name in sorted(_REGISTRY.keys()):
        try:
            b = _REGISTRY[name]()
            infos.append(b.info)
        except Exception:
            infos.append(BackendInfo(name=name, model_name="(failed to load info)"))
    return infos


# ---------------------------------------------------------------------------
# Auto-discover built-in backends on import
# ---------------------------------------------------------------------------

def _discover_builtins() -> None:
    """Import built-in backend modules so they self-register."""
    import importlib
    for mod_name in [
        "robotsmith.gen.hunyuan3d_backend",
        "robotsmith.gen.trellis2_backend",
        "robotsmith.gen.triposg_backend",
    ]:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            pass


_discover_builtins()
