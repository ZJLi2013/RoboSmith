#!/usr/bin/env python3
"""Apply Genesis ROCm cuda.bindings patch for rigid_solver.py"""
import glob, site

candidates = [
    "/opt/venv/lib/python3.12/site-packages/genesis/engine/solvers/rigid/rigid_solver.py",
    "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/genesis/engine/solvers/rigid/rigid_solver.py",
]
for sp in site.getsitepackages():
    candidates += glob.glob(sp + "/genesis/engine/solvers/rigid/rigid_solver.py")

fpath = None
for c in candidates:
    try:
        with open(c) as f:
            f.read(1)
        fpath = c
        break
    except FileNotFoundError:
        continue

if not fpath:
    raise FileNotFoundError("rigid_solver.py not found")

with open(fpath) as f:
    code = f.read()

OLD = '                elif gs.device.type == "cuda":\n                    from cuda.bindings import runtime  # Transitive dependency of torch CUDA\n\n                    _, max_shared_mem = runtime.cudaDeviceGetAttribute(\n                        runtime.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlockOptin, gs.device.index\n                    )\n                    max_shared_mem /= 1024.0'

NEW = '                elif gs.device.type == "cuda":\n                    try:\n                        from cuda.bindings import runtime\n                        _, max_shared_mem = runtime.cudaDeviceGetAttribute(\n                            runtime.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlockOptin, gs.device.index\n                        )\n                        max_shared_mem /= 1024.0\n                    except (ImportError, Exception):\n                        max_shared_mem = 64.0'

if OLD in code:
    code = code.replace(OLD, NEW)
    with open(fpath, 'w') as f:
        f.write(code)
    print(f"Patched {fpath}")
else:
    print(f"Already patched or different version at {fpath}")
