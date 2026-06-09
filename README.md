# LLM-Powered Document Analytics Engine

> **Internship Project** — AI platform that reduces manual document analysis time by ~60% through automated NLP processing, entity extraction, topic modeling, knowledge graph generation, and interactive analytics.

---

## 📋 Project Info

| Field | Value |
|---|---|
| **Project** | LLM-Powered Document Analytics Engine |
| **Student** | Your Full Name |
| **Email** | your.email@university.edu |
| **Stack** | FastAPI · spaCy · NetworkX · React 18 · Vite · Tailwind CSS v4 |

---

## 🗂️ Project Structure

```
doc_analytics/
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI app — all 9 endpoints
│   ├── core/
│   │   ├── document_processor.py  # Module 1 — PDF/DOCX/TXT extraction
│   │   ├── index_builder.py       # Module 2 — Inverted index + TF-IDF search
│   │   ├── entity_extractor.py    # Module 3 — spaCy NER
│   │   ├── topic_extractor.py     # Module 4 — BERTopic + keyword TF-IDF
│   │   ├── graph_builder.py       # Module 5 — NetworkX knowledge graph
│   │   └── analytics.py           # Module 7 — Cross-document aggregation
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── LandingPage.tsx    # Hero + upload + feature cards
│       │   ├── DashboardPage.tsx  # Recharts analytics dashboard
│       │   ├── SearchPage.tsx     # TF-IDF search + highlighted results
│       │   └── KnowledgeGraphPage.tsx # React Flow interactive graph
│       ├── components/
│       │   ├── layout/            # Layout, Sidebar, Header
│       │   └── ui/                # UploadZone, StatCard
│       ├── services/api.ts        # Axios API layer
│       └── types/index.ts         # TypeScript definitions
├── data/
│   ├── uploads/                   # Raw uploaded files
│   ├── processed/                 # doc_id.json — extracted text & pages
│   ├── indexes/                   # doc_id_index.json — inverted search index
│   ├── entities/                  # doc_id_entities.json — NER results
│   ├── topics/                    # doc_id_topics.json — topic model results
│   └── graphs/                    # doc_id_graph.json — NetworkX graph
├── attension.pdf                  # Test document (Attention Is All You Need)
├── pyproject.toml
├── requirements.txt
├── start_backend.sh
└── start_frontend.sh
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- [uv](https://astral.sh/uv) (Python package manager)

### 1 — Install Backend Dependencies

```bash
# Using uv (recommended)
uv pip install fastapi "uvicorn[standard]" python-multipart \
  pymupdf pymupdf4llm python-docx spacy networkx scikit-learn numpy

# Download spaCy English model
uv run python -m spacy download en_core_web_sm

# Optional: BERTopic for advanced topic modeling
uv pip install bertopic
```

### 2 — Start Backend

```bash
./start_backend.sh
# OR
uv run uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend runs at: **http://localhost:8000**  
Swagger docs: **http://localhost:8000/docs**

### 3 — Install & Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: **http://localhost:5174**

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload document file |
| `POST` | `/process/{doc_id}` | Run full NLP pipeline |
| `GET` | `/documents` | List all documents |
| `GET` | `/documents/{doc_id}` | Get processed document |
| `GET` | `/search?q=query` | TF-IDF search across all docs |
| `GET` | `/entities` | Aggregated entity analytics |
| `GET` | `/entities/{doc_id}` | Entities for one document |
| `GET` | `/topics` | Aggregated topic analytics |
| `GET` | `/topics/{doc_id}` | Topics for one document |
| `GET` | `/knowledge-graph` | Merged knowledge graph |
| `GET` | `/knowledge-graph/{doc_id}` | Graph for one document |
| `GET` | `/analytics` | Full dashboard analytics |
| `DELETE` | `/documents/{doc_id}` | Delete document + all data |

---

## 🧠 System Modules

### Module 1 — Document Processing
- **PDF**: PyMuPDF + pymupdf4llm (markdown extraction)
- **DOCX**: python-docx with heading-based section detection
- **TXT/MD/RST/CSV**: Chunked into 500-word pages
- Output: `data/processed/{doc_id}.json`

### Module 2 — Searchable Knowledge Base
- Inverted index with TF-IDF scoring
- Per-document and global cross-document index
- No vector DB, no embeddings — pure JSON storage
- Output: `data/indexes/{doc_id}_index.json`

### Module 3 — Entity Extraction
- spaCy `en_core_web_sm` NER
- 16 entity types: PERSON, ORG, GPE, LOC, PRODUCT, DATE, EVENT…
- Frequency counting + page-level location tracking
- Output: `data/entities/{doc_id}_entities.json`

### Module 4 — Topic & Trend Extraction
- BERTopic (when ≥3 substantial paragraphs available)
- TF-IDF keyword extraction fallback
- Per-page keyword trends
- Output: `data/topics/{doc_id}_topics.json`

### Module 5 — Knowledge Graph
- Entity co-occurrence within ±1 page window
- Relationship type detection from context keywords
- NetworkX degree centrality + betweenness centrality
- Output: `data/graphs/{doc_id}_graph.json`

### Module 6 — Search Engine
- TF-IDF inverted index search
- Relevance-ranked results with keyword highlighting
- Returns: doc name, page, paragraph index, snippet, score

### Module 7 — Analytics Dashboard
- Recharts: BarChart, PieChart, AreaChart
- Entity analytics by type (ORG, PERSON, GPE, PRODUCT)
- Topic frequency distribution
- Knowledge graph density metrics

---

## 🎨 Frontend Pages

| Page | Route | Features |
|------|-------|---------|
| **Landing** | `/` | Hero, upload zone, feature cards, tech stack |
| **Dashboard** | `/dashboard` | Stat cards, 5 chart types, documents table |
| **Search** | `/search` | Query input, ranked results, keyword highlighting |
| **Knowledge Graph** | `/graph` | React Flow graph, node inspector, type filter, minimap |

---

## 🧪 Testing with attension.pdf

```bash
# Upload
UPLOAD=$(curl -s -X POST http://localhost:8000/upload \
  -F "file=@attension.pdf")
DOC_ID=$(echo $UPLOAD | python3 -c "import sys,json; print(json.load(sys.stdin)['doc_id'])")

# Process (runs all 5 modules)
curl -X POST "http://localhost:8000/process/$DOC_ID"

# Search
curl "http://localhost:8000/search?q=attention+mechanism"

# Analytics
curl http://localhost:8000/analytics
```

**Expected results for attension.pdf:**
- 15 pages extracted
- 275 unique entities found
- 5 topics discovered
- 274 graph nodes, 200 edges

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Backend | FastAPI | REST API framework |
| PDF | PyMuPDF + pymupdf4llm | Text & markdown extraction |
| DOCX | python-docx | Word document parsing |
| NLP | spaCy `en_core_web_sm` | Named entity recognition |
| Topics | BERTopic + sklearn TF-IDF | Topic modeling |
| Graph | NetworkX | Knowledge graph & centrality |
| Search | Custom inverted index | TF-IDF ranked search |
| Frontend | React 18 + Vite + TypeScript | UI framework |
| Styling | Tailwind CSS v4 | Design system |
| Charts | Recharts | Analytics visualizations |
| Graph UI | React Flow | Interactive graph visualization |
| Icons | Lucide React | UI icons |
| HTTP | Axios | API client |
| Storage | JSON files | No database required |

---

## 📦 Storage Schema

All data stored as JSON on disk — no database required:

```json
// data/processed/{doc_id}.json
{ "doc_id": "7d1f8124", "document_name": "paper.pdf",
  "total_pages": 15, "pages": [{"page": 1, "text": "...", "paragraphs": [...]}] }

// data/entities/{doc_id}_entities.json
{ "entities": [{ "entity": "Transformer", "type": "ORG", "frequency": 20 }] }

// data/topics/{doc_id}_topics.json
{ "topics": [{ "topic": "Attention / Model", "keywords": ["attention", "model", ...] }] }

// data/graphs/{doc_id}_graph.json
{ "nodes": [...], "edges": [...], "analytics": { "total_nodes": 274, ... } }
```
