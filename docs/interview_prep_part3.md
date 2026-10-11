# Traceable Fusion RAG for Algorithms — Complete Interview Preparation Guide

**Part 3 of 4 — Retrieval, Generation, Guardrails & Transparency**

---

# SECTION 8: RETRIEVAL — [src/retrieval/query_transformer.py](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/retrieval/query_transformer.py)

## 8.1 HyDE: Hypothetical Document Embedding

**Paper**: *"Precise Zero-Shot Dense Retrieval without Relevance Labels"* (Gao et al., 2022)

**Problem**: User queries are short and sparse. "fast sorting" produces a weak, diffuse embedding vector that may not point toward actual quicksort implementations in the vector space.

**HyDE Solution**:
1. Ask the LLM to write a **hypothetical answer** to the query (a code snippet)
2. **Embed the hypothetical answer**, not the raw query
3. Search the FAISS index with the hypothetical embedding

The hypothetical document is far more similar to actual documents in the index than the short query itself — because it *is* a document (code), not a question.

```python
HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a C++ algorithm expert.
Given a question about algorithms, write a SHORT hypothetical C++ code snippet
(50-150 words) that would answer the question.
Include: function signature, key logic, complexity comment.
Output ONLY the code snippet, no explanation."""),
    ("human", "Question: {query}")
])

class HyDEChain:
    def __init__(self, groq_api_key: str, cohere_api_key: str = ""):
        self.groq = ChatGroq(
            api_key=groq_api_key,
            model="openai/gpt-oss-120b",
            temperature=0.3,   # slight creativity for diverse hypotheticals
            max_tokens=300,
            request_timeout=12,   # fail fast, don't hang on rate limits
            max_retries=1,
        )

    def invoke(self, input_dict: dict) -> str:
        query = input_dict["query"]
        # Provider 1: Groq
        try:
            result = (HYDE_PROMPT | self.groq | StrOutputParser()).invoke(input_dict)
            if not _is_refusal(result):
                return result
        except Exception as e:
            pass
        # Provider 2: Cohere Command A+ fallback
        try:
            return _call_cohere_hyde(self._cohere_api_key, query)
        except Exception:
            return query  # last resort: use raw query
```

**Why `temperature=0.3` for HyDE (vs `0.0` for generation)?** A slightly higher temperature encourages the LLM to write varied, exploratory hypothetical code. Low temperature for the main generation is critical for reproducibility; HyDE benefits from a touch of diversity because we want the embedding to explore the neighborhood around the query in vector space.

**Why `request_timeout=12`?** Without a timeout, LangChain's HTTP client will retry a rate-limited request for up to 4 minutes using exponential backoff. This would make every query appear to hang silently. 12 seconds is long enough for a legitimate response but short enough to fail fast and fall over to Cohere.

**[_is_refusal()](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/generation/generator.py#114-127) logic:**
```python
def _is_refusal(text: str) -> bool:
    if len(text) < 300:
        return any(p in text for p in ["rate limit", "429", "quota", "I cannot"])
    return "429" in text or text.strip().startswith("Rate limit")
```
Only check short responses for refusal phrases. A 2000-character response containing the word "rate" in code comments should not be flagged. Long responses are almost certainly legitimate algorithm code.

---

# SECTION 9: RETRIEVAL — [src/retrieval/retriever.py](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/retrieval/retriever.py)

## 9.1 [RetrievedChunk](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/retrieval/retriever.py#27-37) Dataclass

```python
@dataclass
class RetrievedChunk:
    document: Document
    dense_score: float = 0.0      # Cosine similarity (0-1)
    bm25_score: float = 0.0       # BM25 term frequency score
    rrf_score: float = 0.0        # Reciprocal Rank Fusion score
    rerank_score: float = 0.0     # Cohere cross-encoder score
    final_rank: int = 0
    retrieval_method: str = ""    # "dense", "bm25", "hybrid"
```

Every [RetrievedChunk](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/retrieval/retriever.py#27-37) carries all four scores. This is the **Transparency principle** — the full score provenance is preserved all the way to the UI, where users can see exactly why each chunk ranked where it did.

## 9.2 Dense Search (FAISS + Cohere Embeddings)

```python
def dense_search(self, query: str, k: int) -> list[tuple[Document, float]]:
    results = self.vectorstore.similarity_search_with_score(query, k=k)
    # FAISS returns L2 distance (lower = more similar)
    # Convert to similarity: 1.0 - (distance / 2.0) → range [0, 1]
    return [(doc, 1.0 - score/2.0) for doc, score in results]
```

**L2 → Similarity conversion**: FAISS with default settings returns L2 (Euclidean) distances. For unit-normalized vectors (which Cohere's embeddings are), L2 distance and cosine distance are directly related: `cosine_similarity = 1 - L2²/2`. The `1 - score/2` conversion is an approximation that maps `[0, 2]` distance to `[0, 1]` similarity with 0 = identical.

**Why `k * 2` candidates?** Dense search retrieves `k*2 = 16` candidates before reranking. More initial candidates → better recall → the reranker has a richer pool to select the best `top_n = 5` from. The retrieval recall vs. latency tradeoff is calibrated here.

## 9.3 BM25 Search

```python
# BM25 stop words — stripped from queries, NOT from the index
_BM25_STOPWORDS = {
    "the", "a", "an", "for", "of", "in", "to", "and", "or",
    "code", "implement", "algorithm", "function", "program",
    "show", "give", "me", "find", "make", ...
}

def bm25_search(self, query: str, k: int) -> list[tuple[Document, float]]:
    query_tokens = [t for t in self._tokenize(query)
                    if t not in self._BM25_STOPWORDS]
    if not query_tokens:
        query_tokens = self._tokenize(query)  # fallback: no filtering
    scores = self.bm25.get_scores(query_tokens)

    # Soft language boost if query specifies a language
    lang_hint = self._extract_language_hint(query)
    if lang_hint is not None:
        _, lang_tag = lang_hint
        for idx, chunk in enumerate(self.all_chunks):
            if chunk.metadata.get("language", "") == lang_tag:
                scores[idx] *= 1.4    # 40% boost for matching language
    ...
```

**Why custom stop words for BM25?** The words "algorithm", "implement", "code", "function" appear in virtually every chunk (especially in the context headers we prepend). BM25 penalizes terms that appear in many documents (IDF factor). These words have near-zero IDF — they don't discriminate. Removing them from the query focuses BM25 on the discriminative terms like "dijkstra", "quicksort", "BFS".

**Why NOT remove stop words from the index?** We might legitimately want to find code comments that say "This function" or "the algorithm". Stop words in the index don't hurt BM25 because BM25's IDF naturally gives them near-zero weight. Only query-side filtering matters.

**Soft language boost (×1.4)**: When the user asks "Dijkstra in Java", we multiply BM25 scores for Java chunks by 1.4. This keeps BM25 diverse (other languages still appear) but surfaces the requested language higher in the candidate list. It's "soft" because it's a multiplier, not a hard filter — if no Java implementation exists, we still get Python results.

## 9.4 Reciprocal Rank Fusion (RRF)

```python
RRF_K = 60  # Standard constant from the original RRF paper

def reciprocal_rank_fusion(self, dense_results, bm25_results):
    """
    RRF(d) = Σ 1/(k + rank(d))

    Dense scores are in [0, 1] (cosine similarity).
    BM25 scores are unbounded (IDF-weighted term counts).
    Cannot add them directly! RRF uses only rank position.
    """
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}

    for rank, (doc, _) in enumerate(dense_results, 1):
        doc_id = doc.metadata.get("source") + str(doc.metadata.get("chunk_index", 0))
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (self.RRF_K + rank)
        docs[doc_id] = doc

    for rank, (doc, _) in enumerate(bm25_results, 1):
        doc_id = doc.metadata.get("source") + str(doc.metadata.get("chunk_index", 0))
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (self.RRF_K + rank)
        docs[doc_id] = doc

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [(docs[id_], scores[id_]) for id_ in sorted_ids[:self.k * 2]]
```

**Why RRF instead of score normalization?** Score normalization (e.g., min-max scaling each list then adding) requires the score distribution to be stable. BM25 scores depend on the corpus size and term frequencies — they change as you add documents. Normalized scores in one run are incomparable to another run. RRF is **scale-invariant** — it only uses rank order, which is stable.

**Why k=60?** The original paper by Cormack et al. (2009) found k=60 empirically works best across many retrieval tasks. It controls how much weight is given to top-ranked documents vs. lower-ranked: larger k = more weight distributed to lower ranks (more diverse fusion). k=60 is now the de facto standard for RRF.

**Document identity**: `source_path + chunk_index` uniquely identifies a chunk across both lists. When the same chunk appears in both dense and BM25 results (a "double hit"), its RRF score gets contributions from both ranks, naturally boosting candidates that score well semantically AND lexically.

## 9.5 Cross-Encoder Reranking (Cohere Rerank v3.0)

```python
def rerank(self, query, candidates):
    """
    Embedding models: encode query and doc INDEPENDENTLY, then compare.
    Cross-encoders: see BOTH query and doc TOGETHER — full attention.
    """
    lang_hint = self._extract_language_hint(query)
    docs_text = [doc.page_content for doc, _ in candidates]

    response = self.cohere_client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs_text,
        top_n=min(self.rerank_top_n * 3, len(candidates)),
    )

    results = []
    for item in response.results:
        doc, rrf_score = candidates[item.index]
        boosted = self._metadata_boost(item.relevance_score, doc, lang_hint, query=query)
        results.append(RetrievedChunk(
            document=doc,
            rrf_score=rrf_score,
            rerank_score=boosted,
            final_rank=0,
            retrieval_method="hybrid+rerank"
        ))

    results.sort(key=lambda c: c.rerank_score, reverse=True)
    results = results[:self.rerank_top_n]
    for i, r in enumerate(results):
        r.final_rank = i + 1
    return results
```

**Why cross-encoders outperform bi-encoders for reranking?** Bi-encoders (embeddings) compress each document into a fixed-size vector independently. They cannot model the interaction between query and document. A cross-encoder reads the query and document *together* — every token in the query attends to every token in the document. For questions like "Dijkstra shortest path weighted graph", the cross-encoder can recognize that `dist[v] = min(dist[v], dist[u] + weight)` in the code is directly the Dijkstra relaxation step. A bi-encoder would just compare "Dijkstra" embedding with "dist array" embedding.

**Why `top_n * 3` candidates sent to the reranker?** Sending only `top_n=5` candidates is risky — the reranker might find that only 3 are genuinely relevant. Sending `top_n * 3 = 15` gives the cross-encoder a larger pool to select from, improving recall at the cost of slightly more API tokens.

## 9.6 Metadata Boost ([_metadata_boost](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/retrieval/retriever.py#289-344))

This is applied after cross-encoder scoring to adjust for signals the model cannot see:

```python
def _metadata_boost(self, rerank_score, doc, lang_hint, query="") -> float:
    score = rerank_score

    # Signal 1: Language match (+0.15 if matches, -0.25 if mismatches)
    if lang_hint is not None:
        exts, lang_tag = lang_hint
        if doc_lang == lang_tag:
            score = min(1.0, score + 0.15)
        elif doc_lang and doc_lang != lang_tag:
            score = max(0.0, score - 0.25)

    # Signal 2: Filename/algorithm name overlap (+0.12/token, capped +0.20)
    score = max(0.0, score + self._symbolic_name_boost(query, doc))

    # Signal 3: Subroutine-use penalty (-0.30)
    score = max(0.0, score + self._subroutine_penalty(query, doc))

    # Signal 4: README/Markdown bonus (+0.08)
    score += self._readme_bonus(doc)

    return score
```

**Each signal solves a specific failure mode:**

**Signal 1 (language):** The cross-encoder scores Java and C++ Dijkstra implementations equally. But if the user asked "Dijkstra in Java", we want Java first. The metadata boost adds +0.15 for Java and -0.25 for C++. Why asymmetric? A mismatch is a stronger negative signal than a match is positive — we penalize wrong-language results more than we reward right-language results.

**Signal 2 (symbolic name boost):** The cross-encoder sees chunk *text*, not filenames. If the user asks for "Floyd-Warshall" but the chunk body contains only the three nested loops (not the algorithm name), the cross-encoder may give a mediocre score. However, if the filename is `FloydWarshall.java`, that's a direct signal. We split PascalCase filenames into tokens (`FloydWarshallSolver` → `floyd warshall solver`) and count overlaps with query tokens.

**Signal 3 (subroutine penalty):** Chunk from `maths/perfect_square.py` uses binary search internally. Query: "binary search". The cross-encoder sees the word "binary_search" and gives a high score. But `perfect_square.py` is NOT a binary search implementation — it just uses binary search as a subroutine. The penalty (-0.30) detects this pattern: query names algorithm X, but the chunk's filename/category is about a completely different domain.

**Signal 4 (README bonus):** For "how does X work?" questions, Markdown explanation chunks are more useful than code-only chunks. A small +0.08 bonus consistently improves context_recall for conceptual questions.

---

# SECTION 10: GENERATION — [src/generation/generator.py](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/generation/generator.py)

## 10.1 Two-Provider Fallback Chain

```python
class AlgoGenerator:
    def __init__(self, groq_api_key, cohere_api_key):
        self.primary_llm = ChatGroq(
            api_key=groq_api_key,
            model="openai/gpt-oss-120b",
            temperature=0,
            max_tokens=4000,
            request_timeout=20,
            max_retries=1,         # one retry then fail fast
        )
        self._cohere_api_key = cohere_api_key
        self._cohere_client = None   # lazy init

    def _call_with_fallback(self, chain_input, prompt) -> tuple:
        providers = [
            ("Groq Llama 3.3 70B", self._call_groq),
            ("Cohere Command A+",   self._call_cohere),
        ]
        last_error = None
        for name, caller in providers:
            try:
                result = caller(chain_input, prompt)
                if self._is_refusal(result):
                    continue    # treat silent refusal as failure
                return result, name
            except Exception as e:
                last_error = e
        raise RuntimeError(f"All LLM providers failed. Last: {last_error}")
```

**Why Groq as primary?** Groq's LPU (Language Processing Unit) hardware runs Llama 3.3 70B at ~400 tokens/second — 10-20× faster than GPU inference. Free tier: 30,000 tokens/minute, 1,000 requests/day. For a RAG system with 4,000-token responses, this means ~7 queries/minute on the free tier.

**Why Cohere Command A+ as fallback?** 128K context window (vs Groq's 32K), optimized for RAG use cases, different API architecture entirely. When Groq is rate-limited, Cohere is almost certainly not (they're separate companies/APIs). This provides genuine provider diversity for resilience.

**Why `max_retries=1` on Groq?** LangChain by default retries failed requests 2-6 times with exponential backoff. If Groq hits rate limits, this means waiting up to 4 minutes before the exception propagates to our fallback logic. Setting `max_retries=1` means one retry (2 total attempts), then fail fast so Cohere fires within seconds.

**Lazy init of Cohere client**: `self._cohere_client = None` initially. [_get_cohere_client()](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/generation/generator.py#75-79) builds it on first use. This avoids importing and connecting to Cohere when Groq is working fine — reducing startup time and unnecessary network connections.

## 10.2 JSON Response Parsing with Progressive Repair

```python
def parse_json_response(self, text: str) -> dict:
    # Step 1: strip markdown fences
    cleaned = re.sub(r'```(?:json|cpp|c\+\+|python|java)?\s*', '', text).strip()

    def _repair_and_parse(s):
        # Direct parse
        try: return json.loads(s)
        except json.JSONDecodeError: pass

        # Repair 1: escape raw newlines/tabs inside strings
        try:
            fixed = re.sub(
                r'"((?:[^"\\]|\\.)*)"',
                lambda m: '"' + m.group(1).replace('\n','\\n').replace('\t','\\t') + '"',
                s, flags=re.DOTALL
            )
            return json.loads(fixed)
        except: pass

        # Repair 2: triple-quote replacement
        try:
            fixed2 = s.replace('"""', '\\"\\"\\"')
            return json.loads(fixed2)
        except: pass
        return None

    # Try whole string
    result = _repair_and_parse(cleaned)
    if result: return result

    # Try extracting outermost { ... } block
    start, end = cleaned.find('{'), cleaned.rfind('}') + 1
    if start != -1:
        result = _repair_and_parse(cleaned[start:end])
        if result: return result

    # Extract code fields manually (regex-based fallback)
    code_blocks = re.findall(r'"code"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned, re.DOTALL)
    ...

    # Detect raw code response (LLM ignored JSON instructions)
    code_indicators = ["#include", "def ", "class ", "public static"]
    if any(ind in cleaned for ind in code_indicators):
        return {"understanding": "Generated code", "algorithms": [{"code": cleaned}]}

    # Unrecoverable — return error dict
    return {"understanding": "Parse error", "algorithms": [], "parse_error": True}
```

**Why 6 levels of progressive repair?** LLMs are instructed to output JSON but frequently fail in predictable ways:
1. Markdown code fences around the JSON (```` ```json ... ``` ````)
2. Raw newlines inside JSON string values (the "code" field contains actual newlines)
3. Triple-quotes inside JSON strings (Python docstrings have `"""`)
4. Preamble text before the JSON (`"Here is the answer: {...}"`)
5. Code blocks instead of JSON (LLM ignored the format instruction)
6. Complete parse failure (return a structured error)

Each repair step handles one failure mode. The progressive nature means we try the cheapest fix first and escalate only when needed.

**The newline repair regex** is the most complex but most important:
```python
re.sub(r'"((?:[^"\\]|\\.)*)"',
       lambda m: '"' + m.group(1).replace('\n','\\n') + '"',
       s, flags=re.DOTALL)
```
This matches each JSON string value and escapes raw newlines within it. The `(?:[^"\\]|\\.)*` matches either a non-quote, non-backslash character OR a backslash followed by anything — correctly handling already-escaped sequences without double-escaping.

## 10.3 Streaming Support

```python
def stream_response(self, chain_input, prompt) -> Generator[str, None, None]:
    try:
        chain = prompt | self.primary_llm
        collected = []
        for chunk in chain.stream(chain_input):
            if isinstance(chunk, AIMessageChunk):
                collected.append(chunk.content)
        full_text = "".join(collected)
        if not self._is_refusal(full_text):
            yield full_text
            return
    except Exception:
        pass
    # Cohere doesn't stream — yield full response
    result = self._call_cohere(chain_input, prompt)
    yield result
```

**Stream then yield whole**: Rather than yielding each token (which would prevent refusal checking), we collect all streamed tokens, check for refusal strings on the full text, then yield. This preserves the refusal detection logic at the cost of true streaming — a deliberate trade-off for reliability.

---

# SECTION 11: GENERATION — [src/generation/prompt_builder.py](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/generation/prompt_builder.py)

## 11.1 The Algorithm Suggestion Prompt

The main prompt enforces seven strict rules:
1. **ONLY** use code from the provided context
2. If context lacks the algorithm, say so explicitly (not hallucinate)
3. **Never write code from memory** — only adapt from context
4. **Always cite** source files with `[File: <filename>]`
5. **COMPLEXITY RULE**: If a `⚠️ COMPLEXITY NOTE:` is in the context, use that complexity — not any O() from comments (because comments can be wrong)
6. **COMPARISON RULE**: Only compare algorithms that appear in the retrieved context
7. **Complete code**: Never truncate — copy verbatim

**Why such strict negative instructions?** LLMs default to being helpful by filling gaps from parametric memory. Without explicit "NEVER write from memory", the LLM will complete a partial quicksort from memory. With it, the LLM says "the context only shows the partition step, I cannot provide the full implementation" — which is the honest, correct answer.

## 11.2 Code-Aware Complexity Inference ([infer_complexity_from_code](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/generation/prompt_builder.py#217-306))

This is one of the most sophisticated features of the system:

```python
def _count_max_loop_depth(code: str) -> int:
    """
    State machine: tracks brace depth and loop stack simultaneously.
    Correctly distinguishes sequential loops (depth 1 each) from
    nested loops (depth 2, 3, ...).
    """
    brace_depth = 0
    loop_stack: list[int] = []
    max_depth = 0

    tokens = re.split(r'(\{|\}|\bfor\b|\bwhile\b)', stripped)
    for tok in tokens:
        if tok == '{':
            brace_depth += 1
        elif tok == '}':
            brace_depth = max(0, brace_depth - 1)
            loop_stack = [d for d in loop_stack if d < brace_depth]
        elif tok in ('for', 'while'):
            loop_stack.append(brace_depth)
            max_depth = max(max_depth, len(loop_stack))
    return max_depth

def infer_complexity_from_code(code, language) -> dict:
    has_heap = any(re.search(p, code, re.IGNORECASE) for p in heap_patterns)
    has_adj_matrix = "graph[u][v]" in code or "dist[i][j]" in code or ...
    has_linear_min = bool(re.search(r'for.*min.*key|minkey', code_lower)) and not has_heap
    loop_depth = _count_max_loop_depth(code)

    if loop_depth >= 3 and not has_heap:
        return {"time_hint": "O(V³)", "note": "Three nested loops — Floyd-Warshall → O(V³)"}
    elif has_adj_matrix and has_linear_min and not has_heap:
        return {"time_hint": "O(V²)", "note": "Matrix + linear scan — O(V²), NOT O(E log V)"}
    elif has_heap and not has_adj_matrix:
        return {"time_hint": "O((V+E) log V)", "note": "Heap + adjacency list"}
    elif has_sort:
        return {"time_hint": "O(n log n) due to sorting", "note": "..."}
    return {"time_hint": None, "note": None}
```

**The critical bug this fixes**: Prim's MST algorithm exists in two variants:
- **Heap-based**: adjacency list + priority queue → O((V+E) log V)
- **Matrix-based**: adjacency matrix + linear scan → O(V²)

Algorithm repositories often contain the matrix variant (simpler to implement) but the comment says `// O(E log V)` — copied from textbook descriptions of the heap variant. The LLM reads this comment and reports the wrong complexity.

The code inspector detects: no heap usage, adjacency matrix subscripts, linear min-scan → infers O(V²) and injects a `⚠️ COMPLEXITY NOTE` into the prompt that overrides the comment.

**Floyd-Warshall O(V²) vs O(V³) bug**: A naive regex checking "if there are two or more [for](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/evaluation/ragas_eval.py#139-201) loops" would incorrectly flag `Floyd-Warshall` (three sequential two-loop sections) as O(V²). The state machine correctly counts that all three loops are *nested* (depth 3) → O(V³).

## 11.3 Context Assembly ([format_context_for_prompt](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/generation/prompt_builder.py#308-381))

```python
def format_context_for_prompt(retrieved_chunks, max_chars_per_chunk=3000) -> str:
    # Group chunks by source file
    file_groups = defaultdict(list)
    for rank, chunk in enumerate(retrieved_chunks):
        src = chunk.document.metadata.get("source", "unknown")
        file_groups[src].append(chunk)

    context_parts = []
    for i, src in enumerate(sorted_files, 1):
        chunks = file_groups[src]
        merged_code = _merge_chunks(chunks)     # de-duplicate overlapping content
        lang = chunks[0].document.metadata.get("language", "cpp")
        complexity_info = infer_complexity_from_code(merged_code, lang)

        header = (
            f"[{i}] File: {src} | Language: {lang} | "
            f"Inferred Complexity: {complexity_display}"
        )
        if complexity_info["note"]:
            header += f"\n⚠️  COMPLEXITY NOTE: {complexity_info['note']}"

        context_parts.append(f"{header}\n```{lang}\n{merged_code}\n```")

    # Numbered source file index at the top
    source_index = "AVAILABLE SOURCE FILES (cite these exactly):\n"
    for i, src in enumerate(sorted_files, 1):
        source_index += f"  [{i}] {src}\n"

    return source_index + "\n---\n\n" + "\n\n---\n\n".join(context_parts)
```

**Chunk merging within a file**: If we retrieved chunks at indices 2 and 3 from `graph/dijkstra.cpp`, they overlap by 200 characters. Sending them separately would show the LLM the same code twice. [_merge_chunks()](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/generation/prompt_builder.py#115-150) detects the overlap and stitches them into one continuous block — the LLM sees the full function, not two fragments.

**Numbered source file index**: By listing all source files at the top with numbers, the LLM can reference `[1] graph/dijkstra.cpp` in its output rather than writing "some file from the graph directory" — a direct prevention of vague source citations.

---

# SECTION 12: GUARDRAILS — [src/guardrails/confidence.py](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/guardrails/confidence.py)

## 12.1 Pre-Retrieval Gate

```python
def pre_retrieval_gate(retrieved_chunks, min_rerank_score: float = 0.25):
    if not retrieved_chunks:
        return False, "No relevant algorithms found in the knowledge base."

    top_score = retrieved_chunks[0].rerank_score
    if top_score < min_rerank_score:
        return False, (
            f"Best match score ({top_score:.2f}) is below confidence threshold ({min_rerank_score}). "
            f"The knowledge base may not contain algorithms relevant to your query."
        )
    return True, ""
```

**Why check the TOP score, not the mean?** If the top chunk scores 0.85 and chunks 2-5 score 0.10, the retrieval found one excellent match. Mean would be ~0.25 — barely above threshold. In practice, if the top match is strong, generating an answer from it is reliable. If the top match is weak, all downstream chunks are weak.

**Why `0.25` (from `low_confidence = 0.40` in app.py)?** The app actually uses `settings.low_confidence` as the threshold (0.40 in config). Below this, the LLM would be guessing — no real match in the knowledge base. Refusing is better than hallucinating.

This is the **"hallucination from nothing"** prevention — the most important safety mechanism in a RAG system.

## 12.2 Multi-Signal [ConfidenceScorer](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/guardrails/confidence.py#35-430)

```python
def assess(self, retrieved_chunks, generated_result, query) -> ConfidenceReport:
    # Signal 1: Retrieval quality (40% weight)
    retrieval_score = self.score_retrieval(retrieved_chunks)

    # Signal 2: Source verification (30% weight)
    cited_files = [algo.get("source_file") for algo in generated_result["algorithms"]]
    sources_ok, source_warnings = self.verify_sources(cited_files)

    # Signal 3: Coverage/Faithfulness (30% weight)
    coverage_score, coverage_ok = self.score_coverage(answer_text, retrieved_chunks)

    # Combine with weights
    if coverage_ok:
        overall = 0.40 * retrieval_score + 0.30 * coverage_score + 0.30 * source_score
    else:
        overall = 0.55 * retrieval_score + 0.45 * source_score  # redistribute 30%

    # Additional penalties
    overall -= incompleteness_penalty    # -0.08 per signal of incomplete code
    overall -= 0.10 if all_codes_short   # code appears truncated
    overall -= 0.05 if mixed_languages   # reranker may have conflated across languages
    overall -= 0.05 if source_mismatch   # cited file != top-retrieved file

    overall = min(overall, 0.95)  # hard cap: never 100% confident
    ...
```

**Why three signals?**

- **Retrieval score alone** can be gamed: the LLM might ignore the retrieved context entirely and answer from memory. High retrieval score + low coverage = hallucination.
- **Source verification alone** is too binary: if the LLM cites the right file but fabricates its content, source verification passes but the answer is still wrong.
- **Coverage alone** requires an LLM call and can fail due to rate limits. It's a secondary signal.

**Hard cap at 0.95**: No RAG system should claim 100% confidence. We have a mixed-language corpus, semantic reranking that might conflate similar-sounding algorithms, and an LLM that might add context from memory. 0.95 is an honest maximum.

## 12.3 Source Verification ([verify_sources](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/guardrails/confidence.py#130-183))

```python
def verify_sources(self, cited_files: list[str]) -> tuple[bool, list[str]]:
    known_lower = {f.lower() for f in self.known_files}
    for raw_cited in cited_files:
        # Strip LLM formatting quirks: "File: graph/dijkstra.cpp" → "graph/dijkstra.cpp"
        cited = raw_cited.strip()
        for prefix in ("File: ", "file: ", "[File: ", "[file: "):
            if cited.startswith(prefix):
                cited = cited[len(prefix):].rstrip("]").strip()
        cited = cited.lstrip("/").strip()

        if len(cited) > 120: continue   # it's a sentence, not a path
        if " " in cited and "/" not in cited: continue  # "No Java file provided"

        normalised = cited.lower()
        exact_match = normalised in known_lower
        partial_match = (any(normalised in f for f in known_lower) or
                         any(f.endswith(normalised) for f in known_lower))

        if not exact_match and not partial_match:
            warnings.append(f"⚠️ Cited file '{raw_cited}' not found — hallucination?")
    return len(warnings) == 0, warnings
```

**Why partial matching?** The LLM might cite `dijkstra.cpp` while the actual path is `graph/dijkstra.cpp`. `endswith()` comparison catches this — it's still a real file, just incompletely cited. Hard exact matching would generate many false hallucination alerts for correct but abbreviated citations.

**Why skip strings > 120 chars?** When the LLM can't find a real file, it sometimes writes a full sentence in the `source_file` field: `"No Java implementation was found in the provided context"`. This is a sentence, not a path. Over-length strings are definitively not file paths.

---

# SECTION 13: TRANSPARENCY — TRACER & VISUALISER

## 13.1 Observer Pattern ([src/transparency/tracer.py](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/transparency/tracer.py))

```python
class PipelineTracer:
    def __init__(self):
        self.current_trace: Optional[RAGTrace] = None
        self._callbacks: list = []

    def on_step(self, callback):
        self._callbacks.append(callback)  # register UI callback

    def begin_step(self, step_type: StepType, description: str) -> TraceStep:
        step = TraceStep(step_type=step_type, description=description)
        return step

    def end_step(self, step, data=None, success=True, error=None) -> TraceStep:
        step.complete(data=data, success=success, error=error)
        self.current_trace.steps.append(step)
        for callback in self._callbacks:
            try: callback(step)   # fire all registered listeners
            except: pass          # UI errors never crash the pipeline
        return step
```

**StepType enum** defines the complete pipeline vocabulary:
```
QUERY_RECEIVED → QUERY_TRANSFORM → DENSE_RETRIEVAL → BM25_RETRIEVAL
→ RRF_FUSION → RERANKING → CONFIDENCE_CHECK → GENERATION → COMPLETE
```

**Why Observer pattern?** The retriever doesn't know about the terminal, and the terminal doesn't know about FAISS. They communicate only through the tracer event bus. This means you could swap the Rich terminal for a web WebSocket stream without changing any pipeline code — just register a different callback.

**Why `try/except` around callbacks?** A rendering bug in the UI should not crash the pipeline. The pipeline's job is to retrieve and generate — rendering is secondary. Swallowing callback exceptions ensures UI failures are isolated. This is defense-in-depth: the core value (the answer) should be delivered even if the decoration (the fancy terminal display) breaks.

## 13.2 Rich Terminal UI ([src/transparency/visualiser.py](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/transparency/visualiser.py))

The visualiser uses the `rich` library to create a sophisticated terminal interface:

```python
def render_pipeline_summary(trace: RAGTrace):
    # Extracts step counts from trace data
    for s in trace.steps:
        if s.step_type == StepType.BM25_RETRIEVAL:
            bm25_count = s.data.get("count", 0)
        elif s.step_type == StepType.DENSE_RETRIEVAL:
            dense_count = s.data.get("count", 0)
        ...

    trace_text = Text()
    trace_text.append(f"📚 BM25 Search  {bm25_count} candidates\n", style="blue")
    trace_text.append(f"⚡ Vector Search {dense_count} candidates\n", style="magenta")
    trace_text.append(f"🔀 Rank Fusion   {fusion_count} candidates\n", style="magenta")
    trace_text.append(f"🎯 Top-K         {rerank_count} chunks\n", style="yellow")

    console.print(Panel(trace_text, title="RETRIEVAL TRACE", border_style="dim cyan"))
```

**Semantic color coding**:
- 🔵 Blue (`C_BM25`) → BM25 keyword search
- 🟣 Magenta (`C_VECTOR`) → Vector/dense search
- 🟡 Yellow (`C_RERANK`) → Cross-encoder reranking
- 🟢 Green (`C_SUCCESS`) → Final answer
- 🔴 Red (`C_ERROR`) → Errors

**Why Rich instead of print()?** [print()](file:///d:/WEB-DEV/Multisource%20Traceable%20RAG%20for%20Algorithmic%20Analysis/Multisource-Traceable-RAG-for-Algorithmic-Reasoning/src/ui/app.py#79-82) can't render structured panels, colored text, markdown, or tables. Rich's `Panel`, `Table`, `Markdown`, and `Text` classes create a professional terminal display that matches the complexity of what the system is doing — making it clear to the user which stage of the pipeline produced which output.

---

*[End of Part 3. Continue reading Part 4: App, Tests, Evaluation & Interview Q&A]*
