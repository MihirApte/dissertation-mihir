import zipfile
import os

zip_path = r"C:\Users\HP\Downloads\rave_results.zip"
extract_to = r"C:\Users\HP\Downloads\rave_results_extracted"

os.makedirs(extract_to, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zf:
    for name in zf.namelist():
        if name.endswith('/'):
            continue
        parts = [p for p in name.split('/') if p]
        flat_name = "__".join(parts)  # keeps full path, so no collisions
        if len(flat_name) > 200:      # safety margin under Windows' 255-char limit
            base, ext = os.path.splitext(flat_name)
            flat_name = base[:190] + ext
        target = os.path.join(extract_to, flat_name)
        with zf.open(name) as src, open(target, 'wb') as dst:
            dst.write(src.read())

print("Done. Re-extract into a fresh folder to avoid mixing with the old partial one.")