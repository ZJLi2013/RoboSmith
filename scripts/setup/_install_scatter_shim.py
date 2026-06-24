"""Install a pure-PyTorch torch_scatter shim (replaces the broken C++ ext).

PTV3 only uses segment_csr, which computes segment-wise reductions
given a CSR-format index pointer. This shim implements it via
torch.scatter_reduce on the native PyTorch ops — no compiled ext needed.
"""
import os
import sysconfig
import textwrap

SITE = sysconfig.get_paths()["purelib"]
PKG_DIR = os.path.join(SITE, "torch_scatter")

# Back up old init, write shim
os.makedirs(PKG_DIR, exist_ok=True)
init_path = os.path.join(PKG_DIR, "__init__.py")
bak_path = init_path + ".bak"
if not os.path.exists(bak_path) and os.path.exists(init_path):
    os.rename(init_path, bak_path)

shim_code = textwrap.dedent("""\
    \"\"\"Pure-PyTorch torch_scatter shim for ROCm (no compiled ext).\"\"\"
    import torch

    __version__ = "2.1.2+rocm_shim"


    def segment_csr(src, indptr, reduce="mean"):
        \"\"\"Segment reduction with CSR index pointer.

        Args:
            src: (N, ...) source tensor, rows sorted by segment.
            indptr: (S+1,) CSR-style pointer where segment i spans
                    src[indptr[i]:indptr[i+1]].
            reduce: "mean", "sum", "min", or "max".
        Returns:
            (S, ...) reduced tensor, one row per segment.
        \"\"\"
        S = indptr.numel() - 1
        starts = indptr[:-1]
        ends = indptr[1:]
        lengths = ends - starts

        out_shape = (S,) + src.shape[1:]
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)

        idx = torch.arange(S, device=src.device)
        seg_ids = idx.repeat_interleave(lengths)

        if reduce == "sum":
            out.scatter_add_(0, seg_ids.unsqueeze(-1).expand_as(src), src)
        elif reduce == "mean":
            out.scatter_add_(0, seg_ids.unsqueeze(-1).expand_as(src), src)
            mask = lengths > 0
            out[mask] /= lengths[mask].unsqueeze(-1).to(out.dtype)
        elif reduce == "max":
            out.fill_(float("-inf"))
            out.scatter_reduce_(
                0, seg_ids.unsqueeze(-1).expand_as(src), src, reduce="amax"
            )
        elif reduce == "min":
            out.fill_(float("inf"))
            out.scatter_reduce_(
                0, seg_ids.unsqueeze(-1).expand_as(src), src, reduce="amin"
            )
        else:
            raise ValueError(f"Unsupported reduce: {reduce}")
        return out


    def scatter(src, index, dim=0, out=None, dim_size=None, fill_value=0, reduce="sum"):
        \"\"\"Scatter with reduction (simplified shim).\"\"\"
        if dim_size is None:
            dim_size = int(index.max()) + 1
        size = list(src.shape)
        size[dim] = dim_size
        if out is None:
            out = torch.full(size, fill_value, dtype=src.dtype, device=src.device)
        idx = index.unsqueeze(-1).expand_as(src) if index.dim() < src.dim() else index
        if reduce == "sum" or reduce == "add":
            out.scatter_add_(dim, idx, src)
        elif reduce == "mean":
            out.scatter_add_(dim, idx, src)
            counts = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
            ones = torch.ones(index.shape, dtype=src.dtype, device=src.device)
            counts.scatter_add_(0, index, ones)
            mask = counts > 0
            if dim == 0:
                out[mask] /= counts[mask].unsqueeze(-1)
        elif reduce == "max":
            out.scatter_reduce_(dim, idx, src, reduce="amax")
        elif reduce == "min":
            out.scatter_reduce_(dim, idx, src, reduce="amin")
        else:
            raise ValueError(f"Unsupported reduce: {reduce}")
        return out
""")

with open(init_path, "w") as f:
    f.write(shim_code)

print("INSTALLED torch_scatter shim (pure PyTorch, segment_csr + scatter)")
print(f"  path: {init_path}")
