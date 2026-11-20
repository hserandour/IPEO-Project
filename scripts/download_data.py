"""Download and unzip the SeasFire data for the wildfire project (ENV-540, project 4).

Usage:
    python scripts/download_data.py            # both datasets
    python scripts/download_data.py --only npy # just the preprocessed tiles
    python scripts/download_data.py --keep-zips --force

Datasets (Google Drive links from moodle_docs/deep_learning_projects.pdf, p. 8):
    seasfire_1deg_subset.zarr  ~1.1 GB  full data cube, 966 x 180 x 360 per variable
    seas_npy                   ~1.2 GB  preprocessed 20x20 tiles, train/val/test + countries.geojson

Only the Python standard library is used.
"""

import argparse
import re
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "raw"

DATASETS = {
    "zarr": {
        "id": "1eN2KYHHlWA0Pb1y4j9t8gBrVEetrRlF_",
        "zip": "seasfire_1deg_subset.zarr.zip",
        "dir": "seasfire_1deg_subset.zarr",
    },
    "npy": {
        "id": "1Z_L2A9MXLGtkHhWRH2XgLLfAo-7wLUmD",
        "zip": "seas_npy.zip",
        "dir": "seas_npy",
    },
}

DRIVE_URL = "https://drive.usercontent.google.com/download"
CHUNK = 1 << 20  # 1 MiB


def drive_download_url(file_id: str) -> str:
    """Return a direct URL, getting past Drive's 'too large to scan' page."""
    params = {"id": file_id, "export": "download"}
    with urllib.request.urlopen(f"{DRIVE_URL}?{urllib.parse.urlencode(params)}") as r:
        if "text/html" not in r.headers.get("Content-Type", ""):
            return r.url  # small file, served directly
        page = r.read().decode("utf-8", errors="replace")
    uuid = re.search(r'name="uuid" value="([^"]+)"', page)
    params["confirm"] = "t"
    if uuid:
        params["uuid"] = uuid.group(1)
    return f"{DRIVE_URL}?{urllib.parse.urlencode(params)}"


def download(file_id: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(drive_download_url(file_id)) as r:
        if "text/html" in r.headers.get("Content-Type", ""):
            sys.exit(f"Google Drive returned a web page instead of {dest.name} "
                     "(quota exceeded or link changed). Try again later or download it by hand.")
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as f:
            while chunk := r.read(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {done / 1e6:7.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def safe_extract(zip_path: Path, out_dir: Path) -> None:
    """Unzip, refusing entries that would land outside out_dir."""
    out_dir = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            target = (out_dir / name).resolve()
            if not target.is_relative_to(out_dir):
                sys.exit(f"Unsafe path in {zip_path.name}: {name}")
        z.extractall(out_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=DATASETS, help="download a single dataset")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output folder (default: {DEFAULT_OUT})")
    p.add_argument("--keep-zips", action="store_true", help="keep the .zip files after extracting")
    p.add_argument("--force", action="store_true", help="re-download even if the data is already there")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for key in [args.only] if args.only else DATASETS:
        ds = DATASETS[key]
        target, zip_path = args.out / ds["dir"], args.out / ds["zip"]

        if target.exists() and not args.force:
            print(f"[skip] {target} already exists (use --force to re-download)")
            continue
        if target.exists():
            shutil.rmtree(target)

        if not zip_path.exists() or args.force:
            print(f"[download] {ds['zip']}")
            download(ds["id"], zip_path)
        print(f"[unzip] {ds['zip']}")
        safe_extract(zip_path, args.out)
        if not args.keep_zips:
            zip_path.unlink()
        print(f"[done] {target}")


if __name__ == "__main__":
    main()
