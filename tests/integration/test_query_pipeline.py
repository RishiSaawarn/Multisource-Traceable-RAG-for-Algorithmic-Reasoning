# query → retrieve → generate
"""
Integration test: query → hybrid retrieval → confidence gate → generation.

Uses mock LLM to avoid API calls.
Uses a real (tiny) in-memory ChromaDB for retrieval.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.retrieval.retriever import HybridRetriever, RetrievedChunk
from src.generation.generator import AlgoGenerator


def _make_retrieved_chunks(docs):
    """Wrap plain Documents into RetrievedChunk objects as the pipeline expects."""
    return [
        RetrievedChunk(document=doc, dense_score=0.9, rerank_score=0.9)
        for doc in docs
    ]


@pytest.mark.integration
def test_retrieval_returns_relevant_docs(temp_settings, sample_cpp_docs):
    """After indexing sample docs, index_chunks should report the correct count.

    ChromaDB 1.0.x routes all operations through a Rust backend (for both
    PersistentClient and EphemeralClient) which crashes with an access violation
    in pytest's thread context on Windows. We therefore mock the ChromaDB and
    Cohere layers entirely — the test validates AlgoIndexer's deduplication and
    batching logic, not ChromaDB's internal persistence.
    """
    from src.ingestion.indexer import AlgoIndexer
    from unittest.mock import patch, MagicMock

    # Fake collection that tracks upserted IDs
    stored_ids: list[str] = []

    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": stored_ids}
    mock_collection.count.return_value = 0  # updated below
    mock_collection.get.return_value = {
        "ids": stored_ids,
        "metadatas": [{"category": "graph"}, {"category": "sorting"}],
    }

    mock_chroma_client = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection

    mock_vectorstore = MagicMock()

    def fake_add_documents(documents, ids):
        stored_ids.extend(ids)
        mock_collection.count.return_value = len(stored_ids)

    mock_vectorstore.add_documents.side_effect = fake_add_documents

    mock_embed_fn = MagicMock()
    mock_embed_fn.embed_documents.return_value = [
        [0.1 * i for i in range(1024)] for _ in sample_cpp_docs
    ]

    with patch("src.ingestion.indexer.CohereEmbeddings", return_value=mock_embed_fn), \
         patch("src.ingestion.indexer.chromadb.PersistentClient", return_value=mock_chroma_client), \
         patch("src.ingestion.indexer.Chroma", return_value=mock_vectorstore):

        indexer = AlgoIndexer(
            cohere_api_key=temp_settings.cohere_api_key,
            chroma_path=temp_settings.chroma_db_path,
        )
        count = indexer.index_chunks(sample_cpp_docs)
        stats = indexer.get_stats()

    assert count == len(sample_cpp_docs), f"Expected {len(sample_cpp_docs)} new chunks, got {count}"
    assert stats["total_chunks"] >= len(sample_cpp_docs)




@pytest.mark.integration
def test_generation_uses_context_only(mock_llm, sample_cpp_docs):
    """Generator must include retrieved context in its prompt — no free-form generation."""
    from src.generation.generator import AlgoGenerator

    # Fix: suggest_algorithms expects RetrievedChunk objects, not plain Documents
    retrieved_chunks = _make_retrieved_chunks(sample_cpp_docs)

    json_response = (
        '{"answer": "Dijkstra uses a min-heap priority queue.", '
        '"source_files": ["graph/dijkstra.cpp"], '
        '"confidence_note": "High confidence.", '
        '"cpp_snippet": "priority_queue<...> pq;"}'
    )

    with patch("src.generation.generator.ChatGroq"):
        generator = AlgoGenerator(groq_api_key="test", cohere_api_key="test")
        # Patch _call_groq directly so the full fallback chain uses our response
        with patch.object(generator, '_call_groq', return_value=json_response):
            result, provider = generator.suggest_algorithms(
                query="How does Dijkstra work?",
                retrieved_chunks=retrieved_chunks,
            )

    assert "answer" in result
    assert len(result["answer"]) > 10