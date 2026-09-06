"""
CONCEPT: Indexing is the OFFLINE phase of RAG.
You do it once (or when data updates), not on every query.

Pipeline:
  Chunks → Batch Embed (Cohere) → Store (ChromaDB) → Persist

Key considerations:
- Batch embedding: send 96 chunks at once to Cohere (API limit)
- Rate limiting: respect Cohere's free tier (100 req/min)
- Idempotency: skip already-indexed documents
- Progress tracking: show progress for long indexing runs
"""

import time
import hashlib
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
import cohere
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class AlgoIndexer:
    """
    Manages embedding and storing algorithm chunks in ChromaDB.
    """
    
    COHERE_BATCH_SIZE = 24  # Cohere's max batch size
    
    def __init__(self, 
                 cohere_api_key: str,
                 chroma_path: str,
                 collection_name: str = "algorithms"):
        
        self.cohere_api_key = cohere_api_key
        self.faiss_path = chroma_path  # Reused argument name for compatibility
        self.collection_name = collection_name
        
        # LangChain Cohere Embeddings wrapper
        # CONCEPT: We use input_type="search_document" for indexing
        # This tells Cohere to optimise the embedding for retrieval
        self.embeddings = CohereEmbeddings(
            cohere_api_key=cohere_api_key,
            model="embed-english-v3.0",
        )
        
        from langchain_community.vectorstores import FAISS
        import os
        
        self.vectorstore = None
        if os.path.exists(os.path.join(self.faiss_path, "index.faiss")):
            try:
                self.vectorstore = FAISS.load_local(self.faiss_path, self.embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                logger.warning(f"Could not load existing FAISS index: {e}")
    
    def _generate_chunk_id(self, chunk: Document) -> str:
        """
        Generate deterministic unique IDs for chunks.
        """
        content_hash = hashlib.md5(chunk.page_content.encode('utf-8')).hexdigest()
        return f"{chunk.metadata.get('source', 'unknown')}_{content_hash}"
    
    def index_chunks(self, chunks: list[Document], force_reindex: bool = False) -> int:
        """
        Embed and store chunks. Returns count of newly indexed chunks.
        """
        from langchain_community.vectorstores import FAISS
        import os

        new_chunks = chunks
        
        logger.info(f"Indexing {len(new_chunks)} chunks...")
        
        # Process in batches
        total_indexed = 0
        batch_size = self.COHERE_BATCH_SIZE
        
        total_batches = (len(new_chunks) + batch_size - 1) // batch_size
        for batch_idx, i in enumerate(range(0, len(new_chunks), batch_size)):
            print(f"Embedding batch {batch_idx + 1}/{total_batches} (chunk {i}/{len(new_chunks)})...")
            batch_chunks = new_chunks[i:i + batch_size]
            
            try:
                if self.vectorstore is None:
                    self.vectorstore = FAISS.from_documents(batch_chunks, self.embeddings)
                else:
                    self.vectorstore.add_documents(documents=batch_chunks)
                total_indexed += len(batch_chunks)
                
                # Rate limiting: Cohere trial = 100K tokens/min.
                # Each batch of 24 chunks ≈ 24 × ~350 tokens ≈ 8,400 tokens.
                # 100K / 8,400 ≈ 11 batches/min max → need ~5.5s between batches.
                # We use 6s to leave a comfortable margin.
                time.sleep(6.0)
                
            except Exception as e:
                logger.error(f"Batch {i//batch_size} failed: {e}")
                # Continue with next batch rather than stopping
                time.sleep(5)
                continue
        
        logger.info(f"✅ Indexed {total_indexed} chunks successfully")
        
        if self.vectorstore:
            import os
            os.makedirs(self.faiss_path, exist_ok=True)
            self.vectorstore.save_local(self.faiss_path)
            
        return total_indexed
    
    def get_stats(self) -> dict:
        """Return statistics about the current index."""
        try:
            count = 0
            if self.vectorstore and hasattr(self.vectorstore, 'index'):
                count = self.vectorstore.index.ntotal
            
            return {
                "total_chunks": count,
                "categories": ["N/A"],
                "num_categories": 0,
                "db_path": self.faiss_path,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[dict] = None
    ) -> list[tuple[Document, float]]:
        """
        CONCEPT: similarity_search_with_score returns (document, score) pairs.
        Score = cosine distance (0 = identical, 2 = opposite for cosine).
        ChromaDB with cosine space: lower score = more similar.
        
        We use this for our confidence scoring system later.
        """
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )


# --- Full pipeline runner ---
def run_full_indexing_pipeline(settings) -> "AlgoIndexer":
    """
    Orchestrates the complete ingestion pipeline across ALL repos:
    Clone → Load → Chunk → Embed → Index

    Reads the repo list from config.ALGO_REPOS so adding a new language/repo
    only requires editing config.py — nothing here changes.
    """
    from src.config import ALGO_REPOS
    from src.ingestion.loader import load_all_repositories
    from src.ingestion.chunker import CppAwareChunker, add_context_header

    print("\n[>>] Starting AlgoRAG Indexing Pipeline")
    print("=" * 50)

    # Step 1: Clone/update + load all repos
    print(f"\n[1/3] Cloning & loading {len(ALGO_REPOS)} repo(s)...")
    docs = load_all_repositories(ALGO_REPOS)
    print(f"   Total documents loaded: {len(docs)}")

    # Step 2: Chunk
    print("\n[2/3] Chunking documents...")
    chunker = CppAwareChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    chunks = chunker.chunk_documents(docs)
    chunks = [add_context_header(c) for c in chunks]
    print(f"   Created {len(chunks)} chunks with context headers")

    # Step 3: Embed & Index
    print("\n[3/3] Embedding & indexing (this takes a few minutes)...")
    indexer = AlgoIndexer(
        cohere_api_key=settings.cohere_api_key,
        chroma_path=settings.faiss_db_path,
    )

    new_count = indexer.index_chunks(chunks)

    stats = indexer.get_stats()
    print("\n[OK] Indexing Complete!")
    print(f"   New chunks indexed : {new_count}")
    print(f"   Total in DB        : {stats['total_chunks']}")
    print(f"   Categories         : {', '.join(stats['categories'][:8])}...")
    print(f"   DB stored at       : {stats['db_path']}")

    return indexer


if __name__ == "__main__":
    from src.config import settings
    indexer = run_full_indexing_pipeline(settings)
    
    # Quick test query
    print("\n[?] Test query: 'quicksort algorithm'")
    results = indexer.similarity_search_with_score("quicksort algorithm", k=3)
    for doc, score in results:
        print(f"  Score: {score:.4f} | {doc.metadata['source']} | {doc.metadata.get('function_name','')}")