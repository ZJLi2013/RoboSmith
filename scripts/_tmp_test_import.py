import numpy
print(f"numpy: {numpy.__version__}")
import sys
sys.path.insert(0, "/data/shared/Hunyuan3D-2.1/hy3dshape")
sys.path.insert(0, "/data/shared/Hunyuan3D-2.1")
try:
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    print("hy3dshape import OK")
except Exception as e:
    print(f"hy3dshape import FAILED: {e}")
    import traceback
    traceback.print_exc()
