from pydantic import BaseModel


class ProcessResponse(BaseModel):
    doc_id: str
    num_pages: int
    main_content_pdf: str
    main_content_markdown: str
    pages: list[dict]


class BatchRequest(BaseModel):
    input_dir: str
    output_dir: str | None = None


class BatchResponse(BaseModel):
    job_id: str
    num_files_queued: int


class JobStatus(BaseModel):
    job_id: str
    status: str  # "queued" | "running" | "done" | "error"
    processed: int
    total: int
    errors: list[str] = []
