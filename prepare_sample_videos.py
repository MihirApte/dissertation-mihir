"""
prepare_sample_videos.py
=========================
Looks inside sample_videos/ for anything you've downloaded from Pexels/
Pixabay/Mixkit/etc., and copies it into data/mp4_videos/ ready for RAVE:

  - if a video is longer than 10 seconds -> trims to the first 10s
  - if a video is 10 seconds or shorter  -> copied over untouched

Safe to re-run: anything already present in data/mp4_videos/ is skipped,
so you can keep adding new downloads to sample_videos/ and re-run this
whenever you've grabbed a new batch.

USAGE (run from inside the RAVE project folder, where data/mp4_videos/
already exists):
    python3 prepare_sample_videos.py

Requires ffmpeg + ffprobe on PATH (check with: ffprobe -version).
"""

import os
import shutil
import subprocess

SAMPLE_DIR = "sample_videos"
OUT_DIR = os.path.join("data", "mp4_videos")
MAX_SECONDS = 10
VIDEO_EXTS = (".mp4", ".mov", ".webm", ".avi", ".mkv")


def get_duration(path):
    """Return video duration in seconds via ffprobe, or None if it fails."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"  could not read duration ({e})")
        return None


def trim_video(src, dst, seconds=MAX_SECONDS):
    """Re-encode the first `seconds` of src into dst (clean cut, no artifacts)."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-t", str(seconds),
         "-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart", dst],
        check=True, capture_output=True,
    )


def main():
    if not os.path.isdir(SAMPLE_DIR):
        print(f"'{SAMPLE_DIR}/' not found - create it and put your downloaded "
              f"videos in there first.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(VIDEO_EXTS))
    if not files:
        print(f"No video files found in {SAMPLE_DIR}/")
        return

    print(f"Found {len(files)} video(s) in {SAMPLE_DIR}/\n")

    for fname in files:
        src = os.path.join(SAMPLE_DIR, fname)
        dst = os.path.join(OUT_DIR, fname)

        if os.path.exists(dst):
            print(f"[SKIP] {fname} - already in {OUT_DIR}/")
            continue

        duration = get_duration(src)
        if duration is None:
            print(f"[FAIL] {fname} - could not read duration, check the file manually")
            continue

        try:
            if duration > MAX_SECONDS:
                trim_video(src, dst, MAX_SECONDS)
                print(f"[TRIM] {fname}  ({duration:.1f}s -> {MAX_SECONDS}s)")
            else:
                shutil.copy2(src, dst)
                print(f"[COPY] {fname}  ({duration:.1f}s, untouched)")
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {fname} - ffmpeg error: {e}")

    print("\nDone. Check data/mp4_videos/ for the results.")


if __name__ == "__main__":
    main()
