"""
Run this directly for large-scale offline batches -- it's what you actually
want for "millions of PDFs", not POST'ing them one-by-one to FastAPI.

Usage:
    python scripts/batch_runner.py /path/to/input_dir /path/to/output_dir [num_workers]

IMPORTANT -- model loading and GPU memory:
Each multiprocessing worker below calls process_document() independently,
which means EACH WORKER PROCESS loads its own copy of PaddleOCR-VL and the
signature detector into memory the first time it touches a file (Python
multiprocessing workers don't share the module-level model globals in
ocr_engine.py/signature_detector.py -- those are only cached *within* one
process). On a single GPU this means num_workers copies of the VLM in VRAM
at once, which will OOM fast. Pick ONE of:
  1. Set num_workers to whatever actually fits in your VRAM (often 1-2 for
     a 0.9B VLM on a single consumer GPU -- check `nvidia-smi` while running).
  2. CPU-only: num_workers can be cpu_count()-1 as below; slower per-page but
     no VRAM ceiling.
  3. (Best for real scale) Run PaddleOCR-VL behind a vLLM/FastDeploy serving
     endpoint (see ../README.md) and point ocr_engine.py's vl_rec_backend at
     it -- then every worker process shares ONE model server instead of
     loading its own copy, and num_workers can be much higher.

For multi-machine scale beyond what one box can do, swap the
multiprocessing.Pool below for Celery workers reading from the same file
list (see ../docker-compose.yml for a Redis+worker starting point) -- the
per-file logic doesn't change, only how it's dispatched.
"""
import sys
import time
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import process_document  # noqa: E402
from app import config  # noqa: E402


def _process_one(path: str) -> tuple[str, str | None]:
    try:
        process_document(path)
        return (path, None)
    except Exception as e:
        return (path, str(e))


def run(input_dir: str, output_dir: str | None = None, num_workers: int | None = None):
    if output_dir:
        config.OUTPUT_DIR = output_dir
    files = [str(p) for p in Path(input_dir).iterdir()
             if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}]
    print(f"Found {len(files)} files. Output -> {config.OUTPUT_DIR}")

    # Default is conservative (see module docstring): each worker loads its
    # own model copy, so don't default to cpu_count()-1 on a GPU box.
    num_workers = num_workers or 1
    print(f"Using {num_workers} worker process(es). Raise this only after "
          f"checking it fits your VRAM (nvidia-smi) or once you've moved "
          f"the VLM behind a shared vLLM/FastDeploy server.")
    t0 = time.time()
    errors = []
    with Pool(num_workers) as pool:
        for i, (path, err) in enumerate(pool.imap_unordered(_process_one, files), start=1):
            if err:
                errors.append((path, err))
                print(f"[{i}/{len(files)}] FAILED {path}: {err}")
            elif i % 50 == 0 or i == len(files):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"[{i}/{len(files)}] {rate:.1f} pages/sec, "
                      f"ETA {(len(files) - i) / rate / 60:.1f} min" if rate > 0 else "")

    print(f"Done. {len(files) - len(errors)} ok, {len(errors)} failed in {time.time() - t0:.0f}s")
    if errors:
        Path("batch_errors.log").write_text(
            "\n".join(f"{p}\t{e}" for p, e in errors), encoding="utf-8")
        print("See batch_errors.log for details.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    _output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    _num_workers = int(sys.argv[3]) if len(sys.argv) > 3 else None
    run(sys.argv[1], _output_dir, _num_workers)
