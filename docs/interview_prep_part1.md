# Traceable Fusion RAG for Algorithms — Complete Interview Preparation Guide

**Part 1 of 4 — Project Overview, Architecture & Configuration**

---

# SECTION 1: PROJECT OVERVIEW & MOTIVATION

## 1.1 What Is This Project?

**Traceable Fusion RAG for Algorithms** (also called *AlgoRAG* or *TRACE*) is a Retrieval-Augmented Generation system designed to answer natural-language questions about algorithm implementations across four programming languages — C++, Python, Java, and Markdown documentation.

The system allows you to ask:
- *"How does Dijkstra's algorithm find the shortest path?"*
- *"What is the difference between merge sort and quicksort?"*
- *"Show me a segment tree implementation in Java."*

And it returns **structured, cited answers** grounded in **real, open-source algorithm repositories** — not in the LLM's parametric memory. Every answer cites exact source files, carries a multi-signal confidence score, and shows a full pipeline trace so you can see exactly how it arrived at its answer.

---

## 1.2 The Core Problem RAG Solves

Large Language Models (LLMs) have two fundamental limitations for answering algorithm questions:

1. **Hallucination**: LLMs generate plausible-sounding answers from memory but may fabricate incorrect time complexities, wrong algorithm properties, or code that doesn't match any real implementation.
2. **Staleness**: LLMs have a training cutoff and cannot reference newly added algorithms or implementation-specific details.

Standard RAG solves this by grounding answers in a retrieved knowledge base. But standard RAG still has problems:
- A simple vector search may miss exact-match queries ("BFS" → finds "DFS" because it's semantically similar).
- Retrieved chunks may be from wrong files or wrong languages.
- No transparency about how confident the system is, or which files contributed to the answer.

**This project's innovation** is a **hybrid multi-signal retrieval pipeline** with full traceability — combining BM25 + dense vector search + cross-encoder reranking + HyDE query expansion + code-aware complexity inference + multi-signal confidence scoring.

---

## 1.3 Knowledge Base

The system ingests **6 open-source algorithm repositories** (~1 GB total):

| Repository | Language | Files | Why Chosen |
|---|---|---|---|
| `TheAlgorithms/C-Plus-Plus` | C++ | ~370 | Original corpus, canonical C++ implementations |
| `TheAlgorithms/Python` | Python | ~900 | Largest collection, same directory structure |
| `TheAlgorithms/Java` | Java | ~600 | Java standard implementations |
| `keon/algorithms` | Python | ~200 | Better docstrings, cleaner complexity annotations |
| `williamfiset/Algorithms` | Java | ~300 | Best graph/DP quality, competitive-programmer-written |
| `trekhleb/javascript-algorithms` | Markdown | ~many | Indexed as **Markdown** (not JS!) — best human-written algorithm explanations on GitHub |

**Key design decision: `trekhleb` indexed as Markdown.** This repo's per-algorithm README files contain the best conceptual explanations anywhere on GitHub. The JS code itself is unremarkable, but the docs are exceptional. By indexing only the [.md](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/README.md) files and treating them as `language: "markdown"`, the `/explain` mode retrieves conceptual prose rather than code, dramatically improving explanation quality.

---

## 1.4 How To Run The Project

```bash
# 1. Clone and set up environment
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 2. Set API keys in .env
GROQ_API_KEY=gsk_...
COHERE_API_KEY=...

# 3. Ingest knowledge base (30–90 minutes first time, fully idempotent)
python -m src.ingestion.indexer

# 4. Launch the interactive REPL
python -m src.ui.app
```

---

# SECTION 2: SYSTEM ARCHITECTURE

## 2.1 Full Pipeline Diagram

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  HyDE Query Transformer  (query_transformer.py)             │
│  LLM writes a hypothetical code snippet matching the query  │
│  → embeds that snippet (not the raw query) for dense search │
└───────────────────────┬─────────────────────────────────────┘
                        │ HyDE-expanded query
          ┌─────────────┴──────────────┐
          ▼                            ▼
  Dense Retrieval               BM25 Retrieval
  FAISS + Cohere                rank-bm25, exact
  embed-english-v3.0            keyword matching
  (semantic similarity)         (term frequency)
          │                            │
          └─────────────┬──────────────┘
                        ▼
           Reciprocal Rank Fusion (RRF)
           RRF(d) = Σ 1 / (k=60 + rank)
           Merges both ranked lists by position
                        │
                        ▼
             Cohere Reranker (rerank-english-v3.0)
             Cross-encoder: sees BOTH query + doc
             Much more accurate than dot product
                        │ + metadata_boost() applied
                        │   (language match, filename overlap,
                        │    subroutine penalty, README bonus)
                        ▼
           ┌─────────────────────────────┐
           │  Pre-Retrieval Gate         │
           │  (guardrails/confidence.py) │
           │  top_score < 0.40 → REFUSE  │
           └────────────┬────────────────┘
                        │
                        ▼
           Context Assembly (prompt_builder.py)
           - Chunks from same file are MERGED
           - Code structure inspected for complexity
           - ⚠️ COMPLEXITY NOTE injected if needed
                        │
                        ▼
           ┌────────────────────────────────┐
           │  LLM Generation                │
           │  Primary:  Groq 70B (fast)     │
           │  Fallback: Cohere Command A+   │
           │  Output: structured JSON       │
           └────────────┬───────────────────┘
                        │
                        ▼
           Post-Generation Confidence (confidence.py)
           - retrieval_score (40%)
           - coverage_score / faithfulness (30%)
           - source_verified (30%)
           - Penalties: incompleteness, mixed-lang, mismatch
                        │
                        ▼
           PipelineTracer + Rich Terminal UI (app.py)
           Shows: step timings, retrieval scores,
           confidence breakdown, source citations
```

**Why this pipeline over simpler RAG?** Each component solves a specific failure mode of naive RAG:

| Component | Failure Mode Solved |
|---|---|
| HyDE | Short/vague queries produce weak embedding vectors |
| BM25 | Semantic search misses exact algorithm names ("BFS") |
| RRF | Dense and BM25 scores are incomparable scales |
| Cross-encoder reranking | Dot-product similarity misses query-doc interactions |
| Metadata boost | Language-mismatch retrieval (Java query → C++ results) |
| Pre-retrieval gate | Prevents "hallucination from nothing" |
| Complexity inference | LLM copies generic O() from comments, not actual code structure |
| Coverage scoring | Detects when the answer adds facts not in the retrieved context |

---

## 2.2 Data Flow Summary

```
Offline (one-time):  Repos → load_repository() → chunk_document() → add_context_header()
                     → Cohere embed-english-v3.0 → FAISS index (data/faiss_db/)

Online (per-query):  User query → HyDE → Dense + BM25 → RRF → Rerank → Gate
                     → format_context_for_prompt() → LLM → parse_json_response()
                     → ConfidenceScorer.assess() → Rich terminal display
```

---

## 2.3 Technology Stack & Rationale

| Technology | Role | Why This Choice |
|---|---|---|
| **FAISS** (Facebook AI Similarity Search) | Vector store | Optimised for exact nearest-neighbor search; local, no network; persists to disk |
| **Cohere embed-english-v3.0** | Embeddings | 1024-dim, best-in-class for code retrieval, free tier (100K tokens/min) |
| **rank-bm25** | Sparse retrieval | Pure Python, exact BM25Okapi, no server needed |
| **Cohere rerank-english-v3.0** | Cross-encoder reranking | State-of-the-art, free tier, significantly outperforms bi-encoder reranking |
| **Groq Llama 3.3 70B** | Primary LLM | 30K tokens/min free, ultra-fast inference |
| **Cohere Command A+** | Fallback LLM | 128K context window, RAG-optimised, 1K calls/month free |
| **LangChain** | Orchestration | Standard prompt templates, splitters, streaming |
| **Pydantic BaseSettings** | Configuration | Auto-reads [.env](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/.env), validates types, prevents hardcoded values |
| **Rich** | Terminal UI | Markdown rendering, colored output, panels, tables |
| **GitPython** | Repo management | Programmatic clone/pull, retry logic |
| **RAGAS** | Evaluation | Industry-standard RAG metrics: Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall |

---

# SECTION 3: CONFIGURATION MODULE ([src/config.py](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/config.py))

## 3.1 Design Philosophy: Single Source of Truth

```python
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # API Keys
    groq_api_key: str
    cohere_api_key: str
    langchain_api_key: str = ""

    # LLM Config
    primary_llm: str = "openai/gpt-oss-120b"   # Groq endpoint
    fallback_llm: str = "command-a-03-2025"      # Cohere Command A+
    llm_temperature: float = 0.0
    max_tokens: int = 2048

    # Embedding Config
    embedding_model: str = "embed-english-v3.0"
    embedding_dim: int = 1024

    # Retrieval Config
    retrieval_k: int = 8        # chunks retrieved before reranking
    rerank_top_n: int = 5       # chunks kept after reranking
    similarity_threshold: float = 0.35

    # Confidence Thresholds
    high_confidence: float = 0.80
    medium_confidence: float = 0.60
    low_confidence: float = 0.40

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

**Why Pydantic BaseSettings?**
- Auto-reads from [.env](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/.env) — no manual `os.getenv()` calls scattered everywhere.
- Type validation: if `RETRIEVAL_K=abc` is in [.env](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/.env), it fails fast at startup with a clear error, not silently at query time.
- Single `settings` singleton accessed by all modules — never hardcoded values.
- Adding a new parameter means editing only [config.py](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/config.py), not hunting down every usage site.

**Why `llm_temperature = 0.0`?**
Temperature 0 makes the LLM deterministic. For algorithm questions, we want exact, reproducible answers — not creative variation. A BFS implementation should always be described the same way.

**Why `chunk_overlap = 200` with `chunk_size = 1000`?**
A function body split across two chunks would give the LLM only the first half of the function. Overlap ensures a call-site and its callee's definition appear together in at least one chunk. Set to 20% of chunk_size — standard RAG practice.

## 3.2 The `ALGO_REPOS` List

```python
ALGO_REPOS = [
    {"url": "https://github.com/TheAlgorithms/C-Plus-Plus", "path": "./data/raw/C-Plus-Plus", "language": "cpp"},
    {"url": "https://github.com/TheAlgorithms/Python",      "path": "./data/raw/Python",      "language": "python"},
    {"url": "https://github.com/TheAlgorithms/Java",        "path": "./data/raw/Java",        "language": "java"},
    {"url": "https://github.com/keon/algorithms",           "path": "./data/raw/keon-algorithms", "language": "python"},
]
```

**Key extensibility decision:** Adding a new language or repository requires appending **one dict to this list**. Nothing else changes. The loader, chunker, and indexer all read from `ALGO_REPOS` dynamically. This is the Open/Closed Principle in action — open for extension (add repos), closed for modification (no internal logic changes).

**Why `depth=1` shallow cloning?** We only need the current state of these repos, not the full git history. Shallow cloning reduces bandwidth from ~1 GB per repo to ~200 MB.

---

# SECTION 4: DESIGN PATTERNS USED

## 4.1 Observer / Event Emitter (Tracer)

The [PipelineTracer](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/transparency/tracer.py#74-158) implements the Observer pattern. Every pipeline component receives the tracer and calls `tracer.begin_step()` / `tracer.end_step()`. The UI registers a callback with `tracer.on_step()` and fires rendering logic every time a step completes.

This decouples pipeline logic (retriever, generator) from UI concerns (rendering). The pipeline doesn't know or care about the terminal — it just fires events.

## 4.2 Strategy Pattern (Chunker)

[CppAwareChunker](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/chunker.py#23-203) holds a dict of `_splitters` for each language. At runtime, [_get_splitter(language)](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/chunker.py#136-138) selects the right strategy. Adding a new language = adding one entry to the dict. No if/else chains, no switch statements.

## 4.3 Chain of Responsibility (Provider Fallback)

The generator iterates through a `providers` list: `[("Groq", call_groq), ("Cohere", call_cohere)]`. If Groq fails (exception or refusal), it moves to Cohere. This is a clean chain of responsibility — each provider tries and passes to the next on failure.

## 4.4 Factory Function

[create_hyde_chain()](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/retrieval/query_transformer.py#107-115) is a factory that returns a [HyDEChain](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/retrieval/query_transformer.py#61-105) configured with both API keys. Callers don't construct [HyDEChain](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/retrieval/query_transformer.py#61-105) directly — they use the factory. This hides construction complexity.

## 4.5 Dispatch Table

`_LANGUAGE_CONFIG` in [loader.py](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/loader.py) is a dispatch table mapping language strings to their configuration (extensions, metadata extractor, min_size, extra filter). Adding a new language adds one entry — no conditionals modified.

## 4.6 Dataclass for Value Objects

[RetrievedChunk](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/retrieval/retriever.py#27-37), [TraceStep](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/transparency/tracer.py#30-55), [RAGTrace](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/transparency/tracer.py#57-72), [ConfidenceReport](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/guardrails/confidence.py#23-33) are all `@dataclass`es. They are plain data carriers with no business logic. This follows the principle of separating data from behavior — the business logic lives in the classes that consume these objects.

---

*[End of Part 1. Continue reading Part 2: Ingestion Pipeline]*
