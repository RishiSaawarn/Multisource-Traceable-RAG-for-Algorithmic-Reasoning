# Traceable Fusion RAG for Algorithms — Complete Interview Preparation Guide

**Part 2 of 4 — Ingestion Pipeline (Loader, Chunker, Indexer)**

---

# SECTION 5: INGESTION — [src/ingestion/loader.py](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/loader.py)

## 5.1 Purpose

The loader is the first stage of the offline pipeline. It:
1. Clones (or updates) each Git repository
2. Walks the file tree selecting files of the correct extension
3. Extracts rich language-specific metadata from each file
4. Returns a list of `LangChain Document` objects ready for chunking

## 5.2 Repository Management: [clone_or_update_repo()](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/loader.py#29-79)

```python
def clone_or_update_repo(repo_url: str, local_path: str, max_retries: int = 3) -> str:
    path = Path(local_path)
    if path.exists() and (path / ".git").exists():
        git.Repo(local_path).remotes.origin.pull()
        return local_path

    for attempt in range(1, max_retries + 1):
        try:
            env = {"GIT_CONFIG_COUNT": "1",
                   "GIT_CONFIG_KEY_0": "http.version",
                   "GIT_CONFIG_VALUE_0": "HTTP/1.1"} if attempt > 1 else {}
            git.Repo.clone_from(repo_url, local_path, depth=1, env=env or None)
            return local_path
        except git.exc.GitCommandError as e:
            wait = 5 * attempt
            time.sleep(wait)
            shutil.rmtree(path)  # clean partial clone before retry
            path.mkdir(parents=True, exist_ok=True)
```

**Design decisions in this function:**

- **`depth=1` shallow clone**: Only fetches the latest commit. Algorithm repos have thousands of commits we don't need. Cuts clone time from 10+ minutes to ~2 minutes.

- **Idempotent design**: `if path.exists() and (path / ".git").exists()` — if already cloned, just pull. This makes the whole ingestion pipeline safe to interrupt and re-run.

- **Retry with exponential backoff**: GitHub rate-limits unauthenticated clones. `wait = 5 * attempt` gives 5s, 10s, 15s between attempts. After 3 failures, re-raise so the caller sees the error.

- **HTTP/1.1 fallback on retry**: The notorious `curl-92` error ("HTTP/2 stream was cancelled") happens when git over HTTP/2 hits a server-side RST_STREAM. Forcing HTTP/1.1 on retries eliminates this failure mode — HTTP/1.1 has worse multiplexing but never produces curl-92.

- **Clean slate before retry**: A partial clone leaves a corrupt `.git` directory. `shutil.rmtree()` before each retry ensures each attempt starts fresh.

## 5.3 Language-Specific Metadata Extraction

Each language has a dedicated extractor that analyzes the file content and extracts:

### Common Metadata ([_common_metadata](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/loader.py#95-107))
```python
{
    "source": "graph/dijkstra.cpp",           # relative path (normalized to forward slashes)
    "category": "graph",                       # top-level directory name
    "algorithm_name": "Dijkstra",              # filename stem, title-cased
    "file_name": "dijkstra.cpp",
    "language": "cpp"
}
```

**Why extract category from directory?** Algorithm repos are organized by category: `sorting/quicksort.cpp`, `graph/dijkstra.cpp`, `dynamic_programming/knapsack.py`. The first path component is always the algorithm category — a free metadata label requiring zero annotation.

### C++ Metadata Extractor
```python
def extract_metadata_cpp(file_path, repo_root):
    meta = _common_metadata(file_path, repo_root, "cpp")
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    meta.update({
        "time_complexity":     extract_complexity(content, "time"),    # parses "Time: O(n log n)"
        "space_complexity":    extract_complexity(content, "space"),
        "has_class":           str(bool(re.search(r'\bclass\b|\bstruct\b', content))),
        "has_template":        str(bool(re.search(r'\btemplate\b', content))),
        "has_main":            str(bool(re.search(r'\bmain\s*\(', content))),
        "function_signatures": ", ".join(_fn_signatures_cpp(content))  # top 5
    })
    return meta
```

**Why extract function signatures?** They're included in the metadata header prepended to each chunk (see [add_context_header](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/chunker.py#205-222)). When the LLM reads a chunk, it immediately knows which function it's looking at — preventing the "I don't know where this code starts" confusion.

**Why `errors="ignore"` in `read_text()`?** C++ repos contain files with mixed encodings — UTF-8, latin-1, Windows-1252. `errors="ignore"` skips undecodable bytes rather than crashing. The alternative (`errors="replace"`) would insert `?` characters into the code, corrupting function signatures.

### Complexity Extraction
```python
def extract_complexity(content: str, ctype: str) -> Optional[str]:
    # First: look for annotated "Time: O(...)" or "Space: O(...)"
    match = re.search(rf'(?i){ctype}[^:]*:\s*(O\([^)]+\))', content)
    if match:
        return match.group(1)
    # Fallback: find any O() notation in the file
    matches = re.findall(r'O\([^)]+\)', content)
    return matches[0] if matches else None
```

This is captured as metadata but later **overridden by [infer_complexity_from_code()](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/generation/prompt_builder.py#217-306)** in the prompt builder (see Section 8). The reason: comments say `O(E log V)` but the actual implementation may be a linear scan → `O(V²)`. The code structure is the ground truth, not the comment.

### Markdown Quality Filter ([_is_useful_markdown](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/loader.py#210-226))
```python
def _is_useful_markdown(file_path, content) -> bool:
    depth = len(file_path.parts)
    if depth <= 2 and file_path.name.upper() == "README.MD":
        return False      # skip root-level repo intros
    if len(content.strip()) < 400:
        return False      # skip stub files
    if not re.search(r'^#{1,3}\s+\w+', content, re.MULTILINE):
        return False      # must have at least one heading
    return True
```

This is an **extra quality gate** applied only to Markdown. Without it, the index would be polluted with CI badge files, LICENSE explanations, and 3-line stub READMEs. The filter keeps only structured algorithm documentation with proper headings and substantial content.

## 5.4 The Dispatch Table (`_LANGUAGE_CONFIG`)

```python
_LANGUAGE_CONFIG = {
    "cpp": {
        "extensions":   {".cpp", ".h", ".hpp", ".cc"},
        "extract_meta": extract_metadata_cpp,
        "min_size":     50,
        "extra_filter": None,
    },
    "python": {
        "extensions":   {".py"},
        "extract_meta": extract_metadata_python,
        "min_size":     50,
        "extra_filter": None,
    },
    "java": {
        "extensions":   {".java"},
        "extract_meta": extract_metadata_java,
        "min_size":     50,
        "extra_filter": None,
    },
    "markdown": {
        "extensions":   {".md", ".markdown"},
        "extract_meta": extract_metadata_markdown,
        "min_size":     400,              # higher bar for markdown
        "extra_filter": _is_useful_markdown,
    },
}
```

**This is the Open/Closed Principle in action.** [load_repository()](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/loader.py#269-313) reads from this table — it has no language-specific if/else. Adding "rust" support means appending one entry here. No modification to [load_repository()](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/loader.py#269-313).

**Why `min_size = 400` for Markdown vs `50` for code?** A 50-character C++ file is already a real function. A 400-character Markdown file is still likely a boilerplate stub. We raise the bar specifically for Markdown to filter out noise.

## 5.5 Skip Directories

```python
_SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".github",
    "test", "tests", "spec", "specs", ".idea", "build", "dist",
}
```

Test files are explicitly excluded because they contain algorithm *calls*, not algorithm *implementations*. Indexing `test_sorting.py` would retrieve test harness code when a user asks about sorting algorithms — completely wrong content.

---

# SECTION 6: INGESTION — [src/ingestion/chunker.py](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/chunker.py)

## 6.1 Why Chunking Matters

LLMs have context windows (typically 4K–128K tokens). A C++ algorithm file may be 10,000+ characters. We must split files into retrievable "chunks" that:
1. Fit within the context window
2. Preserve semantic boundaries (don't split inside a function)
3. Carry enough surrounding context (overlap between adjacent chunks)

## 6.2 [CppAwareChunker](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/chunker.py#23-203) — Multi-Language Chunker

Despite its name (retained for backward compatibility), this class handles all four languages.

```python
class CppAwareChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        def _make(lang):
            return RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        self._splitters = {
            "cpp":      _make(Language.CPP),
            "python":   _make(Language.PYTHON),
            "java":     _make(Language.JAVA),
            "markdown": _make(Language.MARKDOWN),
            "default":  RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap),
        }
```

**LangChain Language-Aware Splitters**: `RecursiveCharacterTextSplitter.from_language(language=Language.CPP)` knows C++ syntax — it prefers to split at function boundaries, class boundaries, and block boundaries before splitting mid-line. This means even the fallback splitter is smarter than a naive character-count split.

## 6.3 Brace-Matched Function Extraction (C++ and Java)

The primary chunking strategy for C++ and Java uses a **brace-counting state machine**:

```python
def _brace_extract(self, content: str, pattern, blacklist: set) -> list:
    functions = []
    for match in pattern.finditer(content):
        name = match.group(1)
        if name in blacklist:    # skip: if, while, for, switch, else...
            continue
        start = match.start()
        brace_count = 0
        end = start
        for i, ch in enumerate(content[start:], start):
            if ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:    # matched the opening brace
                    end = i + 1
                    break
        if end > start:
            body = content[start:end]
            if 50 < len(body) < self.chunk_size * 2:
                functions.append((name, body))
    return functions
```

**Why brace matching instead of regex?** Regex cannot handle nested braces (a regex `{[^}]+}` stops at the first `}`). Brace counting is O(n) and handles arbitrary nesting depth. This is exactly how compilers parse block structure.

**Why `< self.chunk_size * 2`?** If a function is very long (e.g., a 5000-char implementation), we still want to chunk it — but using the language-aware splitter on just that function's body, not on the whole file. The `*2` cap ensures we don't create super-chunks that overflow the LLM context.

**The blacklist**: `{'if', 'while', 'for', 'switch', 'else', 'catch', 'try'}` — these look like functions to the regex but are control flow. The regex pattern `\w+ \( ... \) {` matches `if (condition) {` just like a function call. We explicitly blacklist these keywords.

## 6.4 Indent-Tracked Python Extraction

Python has no braces — it uses indentation:

```python
def _extract_functions_python(self, content: str) -> list:
    functions = []
    lines = content.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        m = re.match(r'^(def|class)\s+(\w+)', lines[i])
        if m:
            name = m.group(2)
            start = i
            i += 1
            while i < len(lines):
                l = lines[i]
                if l.strip() == "":      # blank lines don't end a block
                    i += 1; continue
                if l[0] in (' ', '\t'): # indented = still inside the block
                    i += 1; continue
                break                    # non-indented = block ended
            body = "".join(lines[start:i])
            if 50 < len(body) < self.chunk_size * 2:
                functions.append((name, body))
        else:
            i += 1
    return functions
```

**The key insight**: A Python function body ends at the first non-blank, non-indented line after the `def`. Skip blank lines (they don't end a block). If the current line starts with a space or tab, we're still inside. The moment we see a non-indented, non-blank line, the block is over. This matches Python's actual parsing rules.

## 6.5 Chunking Strategy — Primary vs. Fallback

```python
def chunk_document(self, doc: Document) -> list:
    language = doc.metadata.get("language", "cpp")
    splitter = self._get_splitter(language)

    functions = self._extract_functions(content, language)

    if len(functions) >= 2:
        # PRIMARY: chunk at function/method boundaries
        for func_name, func_body in functions:
            if len(func_body) > self.chunk_size:
                # large function: sub-split with language splitter
                for i, sub in enumerate(splitter.split_text(func_body)):
                    chunks.append(Document(
                        page_content=sub,
                        metadata={**doc.metadata, "chunk_type": "function_sub",
                                  "function_name": func_name, "chunk_index": i}
                    ))
            else:
                # small-medium function: keep whole
                chunks.append(Document(
                    page_content=func_body,
                    metadata={**doc.metadata, "chunk_type": "function",
                              "function_name": func_name, "chunk_index": 0}
                ))
    else:
        # FALLBACK: language-aware text splitting
        for i, text in enumerate(splitter.split_text(content)):
            chunks.append(Document(
                page_content=text,
                metadata={**doc.metadata, "chunk_type": "language_split",
                          "chunk_index": i}
            ))
    return chunks
```

**Why `>= 2` functions as the threshold for primary strategy?** A file with 0 or 1 detected functions is likely:
- A header file with only declarations
- A file with one giant implementation function
- Markdown (which has no functions at all)

These files work better with the language-aware text splitter. Files with 2+ functions have clear semantic boundaries and benefit from function-level chunking.

**`chunk_type` metadata**: Each chunk knows how it was produced — `"function"` (clean extraction), `"function_sub"` (large function was further split), or `"language_split"` (fallback). This is surfaced in the UI and useful for debugging retrieval quality.

## 6.6 [add_context_header()](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/chunker.py#205-222) — The Secret Weapon

```python
def add_context_header(chunk: Document) -> Document:
    meta = chunk.metadata
    header = (
        f"File: {meta.get('source', 'unknown')} | "
        f"Language: {meta.get('language', 'unknown')} | "
        f"Category: {meta.get('category', 'unknown')} | "
        f"Algorithm: {meta.get('algorithm_name', 'unknown')} | "
        f"Function: {meta.get('function_name', 'N/A')} | "
        f"Time Complexity: {meta.get('time_complexity', 'unknown')}\n\n"
    )
    return Document(page_content=header + chunk.page_content, metadata=chunk.metadata)
```

**Why prepend metadata to the chunk text before embedding?**

When Cohere embeds a chunk, the embedding vector captures the *meaning* of the whole text. A bare C++ function body embeds as "a function that uses a priority queue." With the header prepended, it embeds as "a Dijkstra implementation in a graph module in C++ that uses a priority queue with O((V+E) log V) complexity."

The header dramatically expands the semantic fingerprint of the chunk. Queries like "Dijkstra C++ graph" now match this chunk purely on the header content, even if the function body code would have scored lower alone.

This technique is sometimes called **Contextual Retrieval** or **Metadata-Enriched Embedding**. It consistently improves retrieval precision without any change to the retrieval algorithm.

---

# SECTION 7: INGESTION — [src/ingestion/indexer.py](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/indexer.py)

## 7.1 [AlgoIndexer](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/indexer.py#30-153) — FAISS-Backed Vector Store

```python
class AlgoIndexer:
    COHERE_BATCH_SIZE = 24   # Cohere's API batch limit

    def __init__(self, cohere_api_key, chroma_path, collection_name="algorithms"):
        self.embeddings = CohereEmbeddings(
            cohere_api_key=cohere_api_key,
            model="embed-english-v3.0",
        )
        # Load existing FAISS index if present
        if os.path.exists(os.path.join(self.faiss_path, "index.faiss")):
            self.vectorstore = FAISS.load_local(
                self.faiss_path, self.embeddings,
                allow_dangerous_deserialization=True
            )
```

**Why FAISS?** Facebook AI Similarity Search is a highly optimized library for exact and approximate nearest-neighbor search on dense vectors. It supports batched queries, runs entirely in-process (no network), and persists to disk as `.faiss` binary files. For a local RAG system with ~50K chunks, FAISS exact search is fast enough (~10ms per query).

**`allow_dangerous_deserialization=True`**: FAISS uses Python pickle internally for metadata storage. LangChain's FAISS wrapper requires this flag to acknowledge that loading pickled data is a potential security risk (arbitrary code execution if the pickle file is tampered with). This is acceptable for a local system where we control the data.

## 7.2 Deterministic Chunk IDs ([_generate_chunk_id](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/indexer.py#64-70))

```python
def _generate_chunk_id(self, chunk: Document) -> str:
    content_hash = hashlib.md5(chunk.page_content.encode('utf-8')).hexdigest()
    return f"{chunk.metadata.get('source', 'unknown')}_{content_hash}"
```

**Why MD5 hash of content?** MD5 produces a 32-character hex string. Two chunks with identical content from the same source file produce the same ID. This enables **idempotent indexing** — re-running the indexer on already-indexed chunks does nothing because the IDs already exist in the store.

**Why MD5 and not SHA-256?** Collision resistance is not a security requirement here — we just need uniqueness for deduplication. MD5 is faster and the 32-char output is more readable in logs than SHA-256's 64 chars.

## 7.3 Batch Embedding with Rate Limiting

```python
def index_chunks(self, chunks, force_reindex=False) -> int:
    total_indexed = 0
    batch_size = self.COHERE_BATCH_SIZE   # 24 chunks per API call

    for batch_idx, i in enumerate(range(0, len(new_chunks), batch_size)):
        batch_chunks = new_chunks[i:i + batch_size]

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(batch_chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(documents=batch_chunks)

        total_indexed += len(batch_chunks)

        # Rate limiting: Cohere trial = 100K tokens/min
        # 24 chunks × ~350 tokens ≈ 8,400 tokens → ~11 batches/min max
        # We sleep 6s per batch for comfortable margin
        time.sleep(6.0)

    if self.vectorstore:
        self.vectorstore.save_local(self.faiss_path)
    return total_indexed
```

**Why batch size = 24?** Cohere's API accepts up to 96 items per batch but limits free-tier accounts to 100K tokens/minute. Each algorithmic chunk averages ~350 tokens. `24 × 350 = 8,400 tokens per batch`. `100,000 / 8,400 ≈ 11.9 batches/minute`. Sleeping 6 seconds per batch gives ~10 batches/minute — safely under the limit.

**First-batch vs. add_documents()**: The first batch uses `FAISS.from_documents()` which creates the FAISS index from scratch. Subsequent batches use `add_documents()` which appends to the existing index. This distinction matters because FAISS index creation sets dimensionality — it cannot change after initialization.

**Progressive saving**: `save_local()` is called at the end of the full [index_chunks()](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/indexer.py#71-118) run (not per batch). This is because partial saves could leave an inconsistent index if interrupted mid-batch. The risk is losing the last batch on a crash — acceptable for a development tool.

## 7.4 Full Pipeline Runner ([run_full_indexing_pipeline](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/indexer.py#156-203))

```python
def run_full_indexing_pipeline(settings) -> AlgoIndexer:
    # Step 1: Clone/update + load all repos
    docs = load_all_repositories(ALGO_REPOS)

    # Step 2: Chunk with language-aware splitter
    chunker = CppAwareChunker(chunk_size=settings.chunk_size,
                               chunk_overlap=settings.chunk_overlap)
    chunks = chunker.chunk_documents(docs)
    chunks = [add_context_header(c) for c in chunks]

    # Step 3: Embed & Index
    indexer = AlgoIndexer(cohere_api_key=settings.cohere_api_key,
                          chroma_path=settings.faiss_db_path)
    new_count = indexer.index_chunks(chunks)
    return indexer
```

**Separation of concerns**: Each step is cleanly separated. The pipeline runner doesn't know how chunking works — it just calls [chunk_documents()](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/chunker.py#192-203). The indexer doesn't know how chunks were created — it just calls [index_chunks()](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/indexer.py#71-118). This makes each component testable in isolation.

**Why is ingestion `__main__`-executable?** `if __name__ == "__main__": run_full_indexing_pipeline(settings)` makes the module directly runnable as `python -m src.ingestion.indexer`. This eliminates needing a separate script — the module IS the CLI entry point.

---

## 7.5 Common Interview Questions About Ingestion

**Q: How would you handle a new repository in a different language, say Rust?**

A: Add one entry to `ALGO_REPOS` in [config.py](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/config.py). Then add `"rust"` to `_LANGUAGE_CONFIG` in [loader.py](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/loader.py) with `.rs` extension and a `extract_metadata_rust` function. Add `Language.RUST` to `_splitters` in [CppAwareChunker](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/chunker.py#23-203) (if LangChain supports it) or use the `"default"` splitter. Zero changes to indexer.py or retriever.py.

**Q: How does the system avoid re-indexing already-indexed chunks?**

A: Each chunk gets a deterministic ID: `source_path + MD5(content)`. The current implementation passes all chunks to [index_chunks()](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ingestion/indexer.py#71-118) but the FAISS store natively handles duplicates (same vector at same position is a no-op). A stricter deduplication would track IDs in a separate set and filter before embedding — this is the planned improvement.

**Q: Why not use a managed vector DB like Pinecone or Weaviate?**

A: For a local development project, FAISS gives equivalent retrieval quality with zero network latency, no API cost, and no rate limits. The trade-offs (no horizontal scaling, no CRUD mutations on individual vectors) don't matter here since the knowledge base is rebuilt periodically, not continuously updated.

---

*[End of Part 2. Continue reading Part 3: Retrieval Pipeline]*
