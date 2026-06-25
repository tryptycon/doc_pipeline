# Document Cleaning & Separation Pipeline (local, open-source, no API key)

Built for: scanned government/legal letters with barcodes, routing stamps,
official seals, signatures and thumbprints, in Hindi + English, typed and
handwritten, at a scale of millions of pages.

This is the **production tier**. A classical-CV (no model download) baseline
was tried first against your 3 sample PDFs — see `BASELINE_FINDINGS.md` for
exactly what it could and couldn't do, and why. The short version: colour
heuristics handle red/purple/blue ink fine, but cannot detect a black-ink
stamp or a black-ink signature sitting next to black-ink text, because there
is no colour difference to key off. That's what the trained models below are
for.

## Chosen stack (open-source, free, runs fully offline once weights are downloaded)

| Stage | Model | Why |
|---|---|---|
| Layout detection (find Stamp/Seal/Table/Text regions) | **PP-DocLayoutV3** (via the `paddleocr`/PaddleX package) | Has a built-in `Stamp` class among 17 layout categories — trained on visual shape, not colour, so it catches black-ink stamps too. Free, Apache-2.0, no API key. |
| Signature detection | **tech4humans/yolov8s-signature-detector** (Ultralytics YOLOv8s, on Hugging Face) | Purpose-built signature detector, MIT-licensed. Fine-tune it on a few hundred of your own labeled pages for best results on Indian signature/thumbprint styles (see `training/`). |
| Main-content transcription (typed + handwritten, Hindi + English) | **PaddleOCR-VL-1.6** (0.9B params) | Current open-weight SOTA on OmniDocBench, explicitly supports Hindi/Devanagari handwriting, has a built-in seal-recognition mode, and is small enough (0.9B) to run at the throughput millions of pages requires. Apache-2.0. |
| Serving | **FastAPI** | Single-document / status-check API. For the actual millions-of-pages run, use `training/../batch_runner.py` (multiprocessing) or the optional Celery workers — pushing a million files through one-at-a-time HTTP calls is the wrong tool. |

Alternatives worth knowing about, if you want to swap something in:
`Qwen2.5-VL-7B` (heavier, very strong general OCR), `DeepSeek-OCR2` /
`GLM-OCR` (very fast, small), `olmOCR-7B` (great at markdown/tables),
`MinerU2.5` (good layout-aware PDF→Markdown). All are open weight, free,
no API key, downloadable from Hugging Face.

## Why this sandbox could only go so far

This conversation's environment has no access to huggingface.co or any
model-weight CDN (it's restricted to PyPI/npm/GitHub for package installs).
So the code below is complete and ready to run, but it was **not**
weight-tested in this session — run it on a machine with normal internet
access (and ideally a GPU; PaddleOCR-VL also runs on CPU, just slower).

## Pipeline flow (per page)

```
PDF/image
   │
   ▼
preprocessing.py      deskew, denoise
   │
   ▼
layout_detector.py    PP-DocLayoutV3 → boxes: Text, Stamp, Table, ...
   │
   ▼
signature_detector.py YOLOv8 signature detector → signature boxes
   │
   ▼
region_extractor.py   crop+save Stamp/Seal/Signature/Thumbprint as PNGs
                       inpaint those regions out of the page image
   │
   ▼
ocr_engine.py          PaddleOCR-VL on the cleaned page → text/markdown
   │
   ▼
storage.py             writes:
                          {doc_id}/main_content.pdf
                          {doc_id}/main_content.md
                          {doc_id}/stamp_1.png, seal_1.png, signature_1.png ...
                          {doc_id}/manifest.json
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# downloads PP-DocLayoutV3 + PaddleOCR-VL weights on first run (needs internet)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API

- `POST /process` — upload one PDF/image, get back the manifest + output paths (synchronous; fine for single documents / a demo / a UI upload box)
- `POST /batch` — point at a directory, enqueues background processing for many files, returns a `job_id`
- `GET /status/{job_id}` — progress
- `GET /health`

## Scaling to millions of documents

`POST /batch` is convenient but a single FastAPI process is not how you'd
push millions of pages through a GPU. For that:

1. Use `scripts/batch_runner.py` directly (no HTTP layer) — it walks a
   directory, splits work across a `multiprocessing.Pool` for CPU-bound
   steps (preprocessing, cropping, PDF assembly) and batches images into the
   GPU model calls (layout + OCR) so the GPU is never idle waiting on I/O.
2. If you need this distributed across more than one machine, swap the
   `multiprocessing.Pool` for Celery workers + Redis (a `docker-compose.yml`
   stub is included) — same task function, just dispatched differently.
3. Run the OCR/layout models behind **vLLM** or **Triton Inference Server**
   once you're past a few hundred thousand pages, so multiple worker
   processes share one batched GPU server instead of each loading its own
   copy of the model into VRAM.

## Fine-tuning on your own documents (`training/`)

The pretrained signature detector and PP-DocLayout model were trained on
generic/Western documents. Indian thumbprint impressions, the specific
"प्राप्त सीएम-2" routing-slip stamp design, and your specific scan quality
are not in their training data. To close that gap:

1. Label ~300–1000 of your pages in [Label Studio](https://labelstud.io)
   (free, open source) with boxes for `stamp`, `seal`, `signature`,
   `thumbprint`. Export as YOLO format.
2. `training/prepare_yolo_dataset.py` — splits into train/val/test, writes
   `dataset.yaml`.
3. `training/train_detector.py` — fine-tunes the YOLOv8s signature/stamp
   detector on your labels (starts from the pretrained checkpoint, so a few
   hundred examples is enough to noticeably improve precision).
4. `training/eval_detector.py` — precision/recall/mAP on your held-out test
   split, so you have a number before trusting it on the full million-page
   run.

PaddleOCR-VL itself generally does **not** need fine-tuning for Hindi — it's
already trained on it — but if accuracy on your specific stamp fonts/seal
designs is weak, PaddleOCR's repo includes LoRA fine-tuning scripts for
exactly that.



tain model by shailendra

python3 training/train_detector.py label_studio_export/data.yaml --epochs 5 --imgsz 1280 --batch 8



fine-tuned weights land in runs/signature_finetune/exp/weights/best.pt.

- python training/eval_detector.py runs/detect/runs/signature_finetune/exp-6/weights/best.pt label_studio_export/data.yaml   




Here's the concrete walkthrough for both pieces — these aren't runnable in this chat sandbox (no GPU/Hugging Face access here), so do this on your own machine after unzipping doc_pipeline_production.zip.
Setup first (once):
bashunzip doc_pipeline_production.zip && cd doc_pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# pick ONE, matching your hardware:
pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/   # GPU, CUDA 12.6
pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/          # CPU
1. Running the batch pipeline
bash# test on a small folder first (2-3 files) before pointing it at millions
python scripts/batch_runner.py /path/to/input_pdfs /path/to/output_dir
What happens: it scans input_pdfs for .pdf/.png/.jpg, processes each through the full pipeline, and writes per-document folders under output_dir:
output_dir/
  some_doc_a1b2c3/
    main_content.pdf       # cleaned pages, artifacts removed
    main_content.md        # transcribed text
    page_1/stamp_1.png, signature_1.png, ...
    manifest.json          # bbox + confidence for every extracted region
It defaults to num_workers=1 on purpose — each worker process loads its own full copy of the model, so on a single GPU, raising this without checking VRAM (nvidia-smi while it runs) is the fastest way to OOM:
bashpython scripts/batch_runner.py /path/to/input_pdfs /path/to/output_dir 1   # safe default
python scripts/batch_runner.py /path/to/input_pdfs /path/to/output_dir 4   # only if VRAM allows
Failures get logged to batch_errors.log instead of killing the whole run — check that file after a big batch.
Once you're past what one machine can do, that's when the README's Celery/Redis (docker-compose up) or vLLM-serving step makes sense — same per-file logic, just dispatched across more machines or sharing one model server.
2. Fine-tuning the stamp/signature detector on your own scans
bashpip install label-studio
label-studio start

In the browser UI: create a project → Object Detection template → define labels stamp, seal, signature, thumbprint → upload ~300–1000 representative pages (mix typed + handwritten, different stamp inks/colors — diversity matters more than raw count here, since you're fine-tuning, not training from scratch).
Label every instance of those 4 things on each page.
Export → YOLO format — gives you a zip with images/, labels/, classes.txt.

Then:
bashpython training/prepare_yolo_dataset.py /path/to/label_studio_export /path/to/dataset_out \
    --val-frac 0.15 --test-frac 0.15
→ writes dataset_out/{train,val,test}/ + dataset_out/dataset.yaml.
bashpython training/train_detector.py /path/to/dataset_out/dataset.yaml --epochs 100 --imgsz 1280
→ fine-tuned weights land in runs/signature_finetune/exp/weights/best.pt.
bashpython training/eval_detector.py runs/signature_finetune/exp/weights/best.pt /path/to/dataset_out/dataset.yaml
→ precision/recall/mAP@0.5 per class on the held-out test split — compare this against the pretrained model's numbers on the same test set so you know fine-tuning actually helped before trusting it on real volume.
One wiring step the README doesn't spell out: to actually use best.pt instead of the stock download, edit app/signature_detector.py — swap the hf_hub_download(...) call for just pointing YOLO() at your local path:
python_model = YOLO("/path/to/runs/signature_finetune/exp/weights/best.pt")
Want me to write that edit into the zip directly, or walk through reading the per-class eval numbers once you have them?