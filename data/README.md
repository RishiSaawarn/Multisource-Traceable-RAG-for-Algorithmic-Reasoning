# data/

This directory is **not committed to git** (see `.gitignore`).

It is created automatically when you run the ingestion pipeline:

```bash
python scripts/ingest.py
```

## Contents after ingestion

```
data/
├── raw/                     ← Cloned GitHub repos (~200 MB)
│   ├── C-Plus-Plus/
│   ├── Python/
│   ├── Java/
│   ├── keon-algorithms/
│   ├── williamfiset-algorithms/
│   └── trekhleb-js-algorithms/
│
├── chroma_db/               ← ChromaDB vector store (~800 MB)
└── chroma_db_backup/        ← Backup snapshot (scripts/backup_chroma.py)
```

## Why not committed?

- `data/raw/` — 6 full GitHub repos (~200 MB of source code). Committing
  third-party repos is both wasteful and legally uncertain.
- `data/chroma_db/` — ChromaDB stores 1024-dimensional float32 vectors for
  every chunk. At ~15,000 chunks × 1024 dims × 4 bytes ≈ 60 MB of raw
  vectors, plus metadata and HNSW graph files = ~800 MB total.

GitHub has a 100 MB per-file hard limit and a ~1 GB soft repo limit, making
this impossible to commit even if we wanted to.
