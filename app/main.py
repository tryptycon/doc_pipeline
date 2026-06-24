import shutil
import uuid
from pathlib import Path
from threading import Thread

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from . import config
from .pipeline import process_document
from .schemas import ProcessResponse, BatchRequest, BatchResponse, JobStatus

app = FastAPI(title="Document Cleaning & Separation Pipeline")

_jobs: dict[str, dict] = {}  # in-memory job tracker; swap for Redis/DB at scale


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process", response_model=ProcessResponse)
async def process(file: UploadFile = File(...)):
    """Synchronous single-document processing. Fine for one-off uploads /
    a UI upload box. For many files at once, use /batch instead."""
    upload_dir = Path("./_uploads")
    upload_dir.mkdir(exist_ok=True)
    tmp_path = upload_dir / f"{uuid.uuid4().hex}_{file.filename}"
    with tmp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        manifest = process_document(str(tmp_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return manifest


@app.post("/batch", response_model=BatchResponse)
def batch(req: BatchRequest):
    """Kicks off background processing for every PDF/image in a directory.
    For true millions-of-pages scale, use scripts/batch_runner.py directly
    instead -- see the README's 'Scaling to millions of documents' section."""
    input_dir = Path(req.input_dir)
    if not input_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"{req.input_dir} is not a directory")

    files = [p for p in input_dir.iterdir()
             if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}]
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "queued", "processed": 0, "total": len(files), "errors": []}

    def run():
        _jobs[job_id]["status"] = "running"
        for f in files:
            try:
                process_document(str(f))
            except Exception as e:
                _jobs[job_id]["errors"].append(f"{f.name}: {e}")
            _jobs[job_id]["processed"] += 1
        _jobs[job_id]["status"] = "done"

    Thread(target=run, daemon=True).start()
    return BatchResponse(job_id=job_id, num_files_queued=len(files))


@app.get("/status/{job_id}", response_model=JobStatus)
def status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="unknown job_id")
    job = _jobs[job_id]
    return JobStatus(job_id=job_id, **job)


@app.get("/download/{doc_id}/{filename}")
def download(doc_id: str, filename: str):
    path = Path(config.OUTPUT_DIR) / doc_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(str(path))
