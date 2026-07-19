"""
patch_basicsr.py
==================
Fixes: ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'

Cause: basicsr's data/degradations.py imports from a torchvision module path
that was removed in newer torchvision versions. Locates basicsr's install
path (without fully importing it, since the broken import means a normal
`import basicsr` fails) and rewrites just that one import line.

Run once after installing basicsr in a fresh environment:
    python patch_basicsr.py
"""
import importlib.util
import os

spec = importlib.util.find_spec("basicsr")
if spec is None or spec.submodule_search_locations is None:
    print("[FAIL] basicsr not found - is it installed?")
else:
    basicsr_path = list(spec.submodule_search_locations)[0]
    target = os.path.join(basicsr_path, "data", "degradations.py")

    if not os.path.exists(target):
        print(f"[FAIL] {target} not found - check basicsr install layout")
    else:
        with open(target, "r") as f:
            content = f.read()

        old = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
        new = "from torchvision.transforms.functional import rgb_to_grayscale"

        if new in content:
            print(f"[SKIP] {target} - already patched")
        elif old not in content:
            print(f"[WARN] {target} - expected import line not found, check manually")
        else:
            content = content.replace(old, new)
            with open(target, "w") as f:
                f.write(content)
            print(f"[OK] {target} - patched")
