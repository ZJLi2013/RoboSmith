"""Convert spconv checkpoint weights from CUDA 5D to spconv_rocm 3D format.

CUDA spconv stores SubMConv3d weights as [out, kx, ky, kz, in].
spconv_rocm stores them as [kernel_volume, in, out].

This script converts all affected weights in GraspGen checkpoints.
"""
import sys
import torch

gen_path = sys.argv[1]
dis_path = sys.argv[2] if len(sys.argv) > 2 else None

def convert_state_dict(state_dict):
    converted = {}
    n_converted = 0
    for key, tensor in state_dict.items():
        if tensor.dim() == 5:
            out_c, kx, ky, kz, in_c = tensor.shape
            K = kx * ky * kz
            new_tensor = tensor.reshape(out_c, K, in_c).permute(1, 2, 0).contiguous()
            converted[key] = new_tensor
            n_converted += 1
            print(f"  {key}: {list(tensor.shape)} -> {list(new_tensor.shape)}")
        else:
            converted[key] = tensor
    return converted, n_converted

print(f"Loading generator checkpoint: {gen_path}")
gen_ckpt = torch.load(gen_path, map_location="cpu", weights_only=False)
model_sd = gen_ckpt["model"]
print(f"  Total keys: {len(model_sd)}")
model_sd_new, n = convert_state_dict(model_sd)
gen_ckpt["model"] = model_sd_new
print(f"  Converted {n} 5D -> 3D weights")

out_gen = gen_path.replace(".pth", "_rocm.pth")
torch.save(gen_ckpt, out_gen)
print(f"  Saved: {out_gen}")

if dis_path:
    print(f"\nLoading discriminator checkpoint: {dis_path}")
    dis_ckpt = torch.load(dis_path, map_location="cpu", weights_only=False)
    if "model" in dis_ckpt:
        dis_sd_new, n = convert_state_dict(dis_ckpt["model"])
        dis_ckpt["model"] = dis_sd_new
        print(f"  Converted {n} 5D -> 3D weights")
    elif "state_dict" in dis_ckpt:
        dis_sd_new, n = convert_state_dict(dis_ckpt["state_dict"])
        dis_ckpt["state_dict"] = dis_sd_new
        print(f"  Converted {n} 5D -> 3D weights")
    else:
        for k in dis_ckpt:
            if isinstance(dis_ckpt[k], dict):
                for kk in list(dis_ckpt[k].keys())[:3]:
                    print(f"  top-level key '{k}', sub-key sample: {kk}")
        print("  No standard model key found, checking raw dict...")
        dis_sd_new, n = convert_state_dict(dis_ckpt)
        dis_ckpt = dis_sd_new
        print(f"  Converted {n} 5D -> 3D weights")

    out_dis = dis_path.replace(".pth", "_rocm.pth")
    torch.save(dis_ckpt, out_dis)
    print(f"  Saved: {out_dis}")

print("\nDone. Update your config to point to the *_rocm.pth files.")
