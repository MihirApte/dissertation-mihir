"""
patch_encode_dtype.py
======================
Fixes: RuntimeError: Input type (float) and bias type (struct c10::Half)
should be the same

Cause: after switching self.dtype to float16 (patch_fp16.py), the VAE's
weights became float16, but encode_imgs() was still building the input
image tensor in float32 (2 * split - 1, no cast) before handing it to
self.vae.encode(). This casts that tensor to self.dtype right before
encoding, matching the now-float16 VAE.

Run once from inside the RAVE project folder:
    python patch_encode_dtype.py
"""
import re

FILES = [
    "pipelines/sd_controlnet_rave.py",
    "pipelines/sd_multicontrolnet_rave.py",
]

OLD = "            image = 2 * split - 1\n"
NEW = "            image = (2 * split - 1).to(self.dtype)\n"

for path in FILES:
    with open(path, "r") as f:
        content = f.read()

    if NEW in content:
        print(f"[SKIP] {path} - already patched")
        continue

    if OLD not in content:
        print(f"[WARN] {path} - expected line not found, check manually")
        continue

    content = content.replace(OLD, NEW)
    with open(path, "w") as f:
        f.write(content)
    print(f"[OK] {path} - patched")

print("\nDone. Retry: python scripts/run_experiment.py configs/baseline_random.yaml")
