"""End-to-end test of robotsmith gen pipeline on GPU node.

Tests: search hit -> return URDF; search miss -> hunyuan3d gen -> URDF -> catalog
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, "/tmp/robotsmith-e2e")

print("=== E2E Test: RoboSmith Pipeline ===\n")

# Step 0: Bootstrap
print("Step 0: Bootstrap built-in assets...")
from robotsmith.assets.builtin import bootstrap_builtin_assets
from robotsmith.assets.library import AssetLibrary

assets_root = Path("/tmp/robotsmith-e2e/assets")
bootstrap_builtin_assets(assets_root)
lib = AssetLibrary(assets_root)
print(f"  Library: {lib}")

# Step 1: Search hit
print("\nStep 1: Search 'cup' (should hit mug_red)...")
results = lib.search("cup")
print(f"  Found: {[a.name for a in results]}")
assert len(results) >= 1 and results[0].name == "mug_red", "Search hit failed!"
print(f"  URDF: {results[0].urdf_path}")
print("  PASS")

# Step 2: Search miss -> generate
print("\nStep 2: Search 'purple dragon figurine' (should miss) -> generate via hunyuan3d...")
results = lib.search("purple dragon figurine")
print(f"  Search results: {[a.name for a in results]}")

if not results:
    print("  No match found. Generating via hunyuan3d...")
    t0 = time.time()
    generated = lib.generate("purple dragon figurine", backend="hunyuan3d", target_size_m=0.12)
    gen_time = time.time() - t0
    print(f"  Generated: {generated.name} in {gen_time:.1f}s")
    print(f"  URDF: {generated.urdf_path}")
    print(f"  Tags: {generated.tags}")
    print(f"  Visual mesh: {generated.visual_mesh}")

    # Verify URDF is valid
    assert generated.urdf_path.exists(), "URDF not created!"
    content = generated.urdf_path.read_text()
    assert "<robot" in content, "Invalid URDF!"
    print("  URDF valid")

    # Verify mesh exists
    assert generated.visual_mesh and generated.visual_mesh.exists(), "Visual mesh missing!"
    print("  Visual mesh exists")

    # Verify it's now in library
    assert lib.get(generated.name) is not None, "Not in library!"
    print("  Registered in library")

    # Verify via PyBullet
    print("\n  PyBullet validation...")
    import pybullet as p
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    plane = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=cid)
    p.createMultiBody(0, plane, physicsClientId=cid)
    try:
        body = p.loadURDF(
            str(generated.urdf_path), [0, 0, 0.3],
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=cid,
        )
        for _ in range(480):
            p.stepSimulation(physicsClientId=cid)
        pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=cid)
        print(f"  PyBullet: loaded OK, final_pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        print("  PASS")
    except Exception as e:
        print(f"  PyBullet FAIL: {e}")
    p.disconnect(cid)
else:
    print("  (search matched — skipping generation)")

# Step 3: Library stats
print(f"\nStep 3: Final library state")
print(f"  Total assets: {len(lib)}")
for a in lib.list_all():
    print(f"    {a.name:25s}  source={a.metadata.source}")

print("\n=== E2E COMPLETE ===")
