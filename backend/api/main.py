"""
FastAPI main application - Document Analytics Platform Backend
"""

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Internal modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.document_processor import (
    list_processed_documents,
    load_processed_document,
    process_document,
)
from backend.core.entity_extractor import (
    aggregate_all_entities,
    extract_entities,
    load_entities,
)
from backend.core.graph_builder import (
    aggregate_all_graphs,
    build_knowledge_graph,
    load_graph,
)
from backend.core.index_builder import (
    build_document_index,
    get_document_tree,
    search_documents,
)
from backend.core.topic_extractor import (
    aggregate_all_topics,
    extract_topics,
    load_topics,
)
from backend.core.analytics import get_platform_analytics

# ─── Directory setup ──────────────────────────────────────────────────────────
BASE_DIR = Path("data")
UPLOADS_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".rst", ".csv"}

for d in [UPLOADS_DIR, PROCESSED_DIR, BASE_DIR / "indexes",
          BASE_DIR / "entities", BASE_DIR / "topics", BASE_DIR / "graphs"]:
    d.mkdir(parents=True, exist_ok=True)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LLM-Powered Document Analytics Engine",
    description="AI platform for document analysis, entity extraction, topic modeling, and knowledge graph generation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helper ───────────────────────────────────────────────────────────────────
def _load_registry() -> dict:
    reg_path = BASE_DIR / "registry.json"
    if reg_path.exists():
        with open(reg_path, "r") as f:
            return json.load(f)
    return {"documents": {}}


def _save_registry(registry: dict):
    reg_path = BASE_DIR / "registry.json"
    with open(reg_path, "w") as f:
        json.dump(registry, f, indent=2)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "LLM Document Analytics Engine",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "POST /upload",
            "POST /process/{doc_id}",
            "GET /documents",
            "GET /documents/{doc_id}",
            "GET /search",
            "GET /entities",
            "GET /entities/{doc_id}",
            "GET /topics",
            "GET /topics/{doc_id}",
            "GET /knowledge-graph",
            "GET /knowledge-graph/{doc_id}",
            "GET /analytics",
        ],
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document file. Returns doc_id for subsequent processing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    doc_id = str(uuid.uuid4())[:8]
    save_path = UPLOADS_DIR / f"{doc_id}{suffix}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Register document
    registry = _load_registry()
    registry["documents"][doc_id] = {
        "doc_id": doc_id,
        "original_name": file.filename,
        "saved_path": str(save_path),
        "format": suffix.lstrip("."),
        "status": "uploaded",
    }
    _save_registry(registry)

    return {
        "success": True,
        "doc_id": doc_id,
        "filename": file.filename,
        "format": suffix.lstrip("."),
        "message": f"File uploaded. Call POST /process/{doc_id} to analyze.",
    }


@app.post("/process/{doc_id}")
async def process_doc(doc_id: str):
    """Run full NLP pipeline on an uploaded document."""
    registry = _load_registry()
    doc_meta = registry["documents"].get(doc_id)
    if not doc_meta:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    file_path = Path(doc_meta["saved_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded file not found on disk")

    try:
        # Module 1: Process document
        registry["documents"][doc_id]["status"] = "processing"
        _save_registry(registry)

        doc_data = process_document(file_path, doc_id)

        # Module 2: Build search index
        index_data = build_document_index(doc_data)

        # Module 3: Extract entities
        entity_data = extract_entities(doc_data)

        # Module 4: Extract topics
        topic_data = extract_topics(doc_data)

        # Module 5: Build knowledge graph
        graph_data = build_knowledge_graph(doc_data, entity_data)

        registry["documents"][doc_id]["status"] = "processed"
        registry["documents"][doc_id]["document_name"] = doc_data["document_name"]
        registry["documents"][doc_id]["total_pages"] = doc_data["total_pages"]
        _save_registry(registry)

        return {
            "success": True,
            "doc_id": doc_id,
            "document_name": doc_data["document_name"],
            "total_pages": doc_data["total_pages"],
            "entities_found": entity_data["unique_entities"],
            "topics_found": topic_data["total_topics"],
            "graph_nodes": graph_data["analytics"]["total_nodes"],
            "graph_edges": graph_data["analytics"]["total_edges"],
            "status": "processed",
        }

    except Exception as e:
        registry["documents"][doc_id]["status"] = "error"
        registry["documents"][doc_id]["error"] = str(e)
        _save_registry(registry)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/documents")
async def list_documents():
    """List all uploaded and processed documents."""
    registry = _load_registry()
    docs = list(registry["documents"].values())
    processed = list_processed_documents()
    processed_ids = {d["doc_id"] for d in processed}

    result = []
    for doc in docs:
        doc_copy = dict(doc)
        doc_copy["is_processed"] = doc["doc_id"] in processed_ids
        result.append(doc_copy)

    return {"documents": result, "total": len(result)}


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get the processed content of a specific document."""
    doc_data = load_processed_document(doc_id)
    if not doc_data:
        raise HTTPException(status_code=404, detail=f"Processed document '{doc_id}' not found")
    # Return without full text for performance
    doc_data.pop("full_text", None)
    doc_data.pop("markdown", None)
    return doc_data


@app.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search across all processed documents."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    results = search_documents(q, top_k=limit)
    tree = get_document_tree()

    return {
        "query": q,
        "total_results": len(results),
        "results": results,
        "document_tree": tree,
    }


@app.get("/entities")
async def get_all_entities():
    """Get aggregated entity analytics across all documents."""
    return aggregate_all_entities()


@app.get("/entities/{doc_id}")
async def get_doc_entities(doc_id: str):
    """Get entities for a specific document."""
    data = load_entities(doc_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Entities for '{doc_id}' not found")
    return data


@app.get("/topics")
async def get_all_topics():
    """Get aggregated topic analytics across all documents."""
    return aggregate_all_topics()


@app.get("/topics/{doc_id}")
async def get_doc_topics(doc_id: str):
    """Get topics for a specific document."""
    data = load_topics(doc_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Topics for '{doc_id}' not found")
    return data


@app.get("/knowledge-graph")
async def get_merged_graph():
    """Get the merged knowledge graph across all documents."""
    return aggregate_all_graphs()


@app.get("/knowledge-graph/{doc_id}")
async def get_doc_graph(doc_id: str):
    """Get the knowledge graph for a specific document."""
    data = load_graph(doc_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Graph for '{doc_id}' not found")
    return data


@app.get("/analytics")
async def get_analytics():
    """Get comprehensive platform analytics for the dashboard."""
    return get_platform_analytics()


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all its processed data."""
    registry = _load_registry()
    if doc_id not in registry["documents"]:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    doc_meta = registry["documents"][doc_id]
    deleted_files = []

    # Remove uploaded file
    up_path = Path(doc_meta.get("saved_path", ""))
    if up_path.exists():
        up_path.unlink()
        deleted_files.append(str(up_path))

    # Remove processed files
    for subdir, suffix in [
        ("processed", ".json"),
        ("indexes", "_index.json"),
        ("entities", "_entities.json"),
        ("topics", "_topics.json"),
        ("graphs", "_graph.json"),
    ]:
        p = BASE_DIR / subdir / f"{doc_id}{suffix}"
        if p.exists():
            p.unlink()
            deleted_files.append(str(p))

    del registry["documents"][doc_id]
    _save_registry(registry)

    return {"success": True, "doc_id": doc_id, "deleted_files": deleted_files}
