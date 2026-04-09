"""Scene generation backends: programmatic (MVP) + SceneSmith (future stub)."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from robotsmith.assets.library import AssetLibrary
from robotsmith.assets.schema import Asset
from robotsmith.scenes.config import SceneConfig, ObjectPlacement


@dataclass
class PlacedObject:
    """An asset placed in a scene with a concrete pose."""

    asset: Asset
    position: list[float]
    rotation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class ResolvedScene:
    """A fully resolved scene: concrete objects with poses, ready for sim loading."""

    config: SceneConfig
    placed_objects: list[PlacedObject] = field(default_factory=list)
    table_asset: Optional[Asset] = None
    plane_asset: Optional[Asset] = None

    def summary(self) -> str:
        lines = [f"Scene: {self.config.name} ({len(self.placed_objects)} objects)"]
        for po in self.placed_objects:
            pos = [round(v, 3) for v in po.position]
            lines.append(f"  {po.asset.name:20s} pos={pos}")
        return "\n".join(lines)


class SceneBackend(ABC):
    """Abstract base class for scene generation backends."""

    @abstractmethod
    def resolve(self, config: SceneConfig, library: AssetLibrary) -> ResolvedScene:
        """Resolve a SceneConfig into a ResolvedScene with concrete assets and poses."""
        ...


class ProgrammaticSceneBackend(SceneBackend):
    """MVP scene backend: resolve object placements from config + library search."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def resolve(self, config: SceneConfig, library: AssetLibrary) -> ResolvedScene:
        placed: list[PlacedObject] = []

        for obj_spec in config.objects:
            assets = library.search(obj_spec.asset_query, top_k=obj_spec.count)
            if not assets:
                print(f"[scene] WARNING: no asset found for query={obj_spec.asset_query!r}")
                continue

            for i in range(obj_spec.count):
                asset = assets[i % len(assets)]

                if obj_spec.fixed_position:
                    pos = list(obj_spec.fixed_position)
                else:
                    lo, hi = obj_spec.position_range
                    pos = [
                        self.rng.uniform(lo[j], hi[j])
                        for j in range(3)
                    ]

                lo_r, hi_r = obj_spec.rotation_range
                rot = [self.rng.uniform(lo_r[j], hi_r[j]) for j in range(3)]

                placed.append(PlacedObject(asset=asset, position=pos, rotation=rot))

        table = library.get("table_simple")
        plane = library.get("plane")

        return ResolvedScene(
            config=config,
            placed_objects=placed,
            table_asset=table,
            plane_asset=plane,
        )


class SceneSmithBackend(SceneBackend):
    """Stub for future SceneSmith integration (LLM -> layout -> Drake SDF).

    Not implemented in MVP. Will accept natural language scene descriptions
    and produce object placement layouts via SceneSmith's LLM planning.
    """

    def resolve(self, config: SceneConfig, library: AssetLibrary) -> ResolvedScene:
        raise NotImplementedError(
            "SceneSmithBackend is a future extension. "
            "Use ProgrammaticSceneBackend for MVP."
        )
