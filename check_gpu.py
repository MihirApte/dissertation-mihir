
import sys

print("=" * 60)
print("RAVE - Environment Check")
print("=" * 60)


print(f"\n[1] Python version: {sys.version}")
assert sys.version_info >= (3, 8), "Python 3.8+ required"
print("[OK] OK")


print("\n[2] Checking PyTorch + CUDA...")
try:
    import torch
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("[FAIL] ERROR: CUDA not available. Check your PyTorch install.")
        sys.exit(1)
    print(f"CUDA version    : {torch.version.cuda}")
    print(f"GPU name        : {torch.cuda.get_device_name(0)}")
    print(f"GPU memory      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute cap     : {cap[0]}.{cap[1]}")
    if cap[0] < 8:
        print("[FAIL] WARNING: Compute capability < 8.0. Some features may not work.")
    else:
        print("[OK] OK")
except ImportError:
    print("[FAIL] ERROR: PyTorch not installed.")
    sys.exit(1)


print("\n[3] Running quick GPU tensor test...")
try:
    x = torch.randn(4, 4).cuda()
    y = torch.randn(4, 4).cuda()
    z = x @ y
    assert z.shape == (4, 4)
    print("[OK] OK")
except Exception as e:
    print(f"[FAIL] ERROR: {e}")
    sys.exit(1)


print("\n[4] Checking torch.amp.autocast (Blackwell fix)...")
try:
    with torch.amp.autocast('cuda'):
        x = torch.randn(2, 2).cuda()
        _ = x * x
    print("[OK] OK")
except Exception as e:
    print(f"[FAIL] ERROR: {e}")
    sys.exit(1)


print("\n[5] Checking diffusers...")
try:
    import diffusers
    print(f"diffusers version: {diffusers.__version__}")
    print("[OK] OK")
except ImportError:
    print("[FAIL] ERROR: diffusers not installed. Run: pip install diffusers==0.18.2")
    sys.exit(1)


print("\n[6] Checking transformers...")
try:
    import transformers
    print(f"transformers version: {transformers.__version__}")
    print("[OK] OK")
except ImportError:
    print("[FAIL] ERROR: transformers not installed.")
    sys.exit(1)


print("\n[7] Checking xformers...")
try:
    import xformers
    print(f"xformers version: {xformers.__version__}")
    print("[OK] OK")
except ImportError:
    print("[FAIL] WARNING: xformers not installed. Memory efficiency will be reduced.")
    print("          Install: pip install xformers==0.0.27")


print("\n[8] Checking OpenCV...")
try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
    print("[OK] OK")
except ImportError:
    print("[FAIL] ERROR: opencv-python not installed.")
    sys.exit(1)


print("\n[9] Checking CLIP (required for semantic shuffle)...")
try:
    import clip
    print(f"clip installed [OK]")
    print("[OK] OK")
except ImportError:
    print("[FAIL] ERROR: openai-clip not installed.")
    print("         Run: pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)


print("\n[10] Checking other dependencies...")
missing = []
for pkg in ["PIL", "numpy", "yaml", "imageio", "torchvision", "tqdm"]:
    try:
        __import__(pkg if pkg != "PIL" else "PIL.Image")
        print(f" {pkg} [OK]")
    except ImportError:
        missing.append(pkg)
        print(f" {pkg} [FAIL] MISSING")

if missing:
    print(f"\n    [FAIL] ERROR: Missing packages: {missing}")
    sys.exit(1)

# Final 
print("\n" + "=" * 60)
print("All checks passed. You are ready to run RAVE.")
print("=" * 60)
print("\nNext step:")
print("  bash run_experiments.sh")
print("=" * 60)
