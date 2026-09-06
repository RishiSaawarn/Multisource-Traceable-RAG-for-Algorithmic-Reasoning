"""
TRACE: Traceable Retrieval & Context Engine
"""

import sys
import os
import asyncio
import logging

logger = logging.getLogger(__name__)
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.table import Table

BANNER = r"""
  _______  _____            _____  ______ 
 |__   __||  __ \    /\    / ____||  ____|
    | |   | |__) |  /  \  | |     | |__   
    | |   |  _  /  / /\ \ | |     |  __|  
    | |   | | \ \ / ____ \| |____ | |____ 
    |_|   |_|  \_/_/    \_\\_____||______|

  Traceable Retrieval & Context Engine
  Hybrid Search • Rank Fusion • Grounded Reasoning
"""

HELP_TEXT = """
[bold cyan]Commands:[/bold cyan]
  [yellow]<query>[/yellow]           Search for algorithms (e.g., "efficient sorting algorithm")
  [yellow]/explain <query>[/yellow]  Teaching mode — detailed explanation
  [yellow]/category <name>[/yellow]  Filter by category (sorting, graph, dp, ...)
  [yellow]/index[/yellow]            Re-index the repository
  [yellow]/stats[/yellow]            Show index statistics
  [yellow]/history[/yellow]          Show recent queries
  [yellow]/help[/yellow]             Show this help
  [yellow]/quit[/yellow] or Ctrl+C   Exit

[bold cyan]Tips & Examples:[/bold cyan]
  • [dim]By Name:[/dim] "dijkstra", "quicksort", "BFS"
  • [dim]By Properties:[/dim] "O(1) space", "stable sort", "in-place"
  • [dim]By Problem:[/dim] "find shortest path", "detect cycle in a graph"
"""

load_dotenv()

# --- Internal imports ---
from src.config import settings
from src.ingestion.loader import clone_or_update_repo, load_cpp_repository
from src.ingestion.chunker import CppAwareChunker, add_context_header
from src.ingestion.indexer import AlgoIndexer
from src.retrieval.retriever import HybridRetriever
from src.retrieval.query_transformer import create_hyde_chain
from src.generation.generator import AlgoGenerator
from src.guardrails.confidence import ConfidenceScorer, pre_retrieval_gate
from src.transparency.tracer import PipelineTracer, StepType
from src.transparency.visualiser import (
    console, render_query_banner, render_step,
    render_pipeline_summary, render_algorithm_results
)

class AlgoRAGApp:
    """Main application class — orchestrates the full RAG pipeline."""
    
    def __init__(self):
        self.retriever = None
        self.generator = None
        self.confidence_scorer = None
        self.hyde_chain = None
        self.query_history = []
        self.all_chunks = []
        self.known_files = set()
        self.total_chunks = 0
        
    def print_banner(self):
        console.print(f"[bold cyan]{BANNER}[/bold cyan]")
        console.print(Panel(HELP_TEXT, border_style="dim cyan", padding=(0, 2)))
    
    def initialize(self):
        """Load or create the index and initialize all components."""
        console.print("  [dim]SYSTEM[/dim]")
        
        faiss_path = Path(settings.faiss_db_path)
        index_exists = faiss_path.exists() and any(faiss_path.iterdir())
        
        if not index_exists:
            console.print("  [bold yellow]●[/bold yellow] FAISS Vector Store      [dim][Indexing...][/dim]")
            self._run_indexing()
        
        from src.config import ALGO_REPOS
        from src.ingestion.loader import load_repository
        all_docs = []
        for repo_cfg in ALGO_REPOS:
            repo_docs = load_repository(repo_cfg["path"], repo_cfg["language"])
            all_docs.extend(repo_docs)
            
        chunker = CppAwareChunker(settings.chunk_size, settings.chunk_overlap)
        self.all_chunks = [add_context_header(c) for c in chunker.chunk_documents(all_docs)]
        self.known_files = {c.metadata.get("source", "") for c in self.all_chunks}
        
        indexer = AlgoIndexer(
            cohere_api_key=settings.cohere_api_key,
            chroma_path=settings.faiss_db_path
        )
        
        stats = indexer.get_stats()
        self.total_chunks = stats.get('total_chunks', len(self.all_chunks))
        
        console.print(f"  [bold green]●[/bold green] FAISS Vector Store      [dim][{self.total_chunks:,} chunks][/dim]")
        console.print(f"  [bold green]●[/bold green] BM25 Keyword Search     [dim][{len(self.all_chunks):,} chunks][/dim]")
        console.print(f"  [bold green]●[/bold green] Embedding Service       [dim][Ready][/dim]")
        console.print()
        
        self.retriever = HybridRetriever(
            vectorstore=indexer.vectorstore,
            all_chunks=self.all_chunks,
            cohere_api_key=settings.cohere_api_key,
            k=settings.retrieval_k,
            rerank_top_n=settings.rerank_top_n,
        )
        
        self.generator = AlgoGenerator(
            groq_api_key=settings.groq_api_key,
            cohere_api_key=settings.cohere_api_key,
        )
        
        self.confidence_scorer = ConfidenceScorer(
            groq_api_key=settings.groq_api_key,
            known_files=self.known_files,
            cohere_api_key=settings.cohere_api_key,
        )
        
        self.hyde_chain = create_hyde_chain(
            groq_api_key=settings.groq_api_key,
            cohere_api_key=settings.cohere_api_key,
        )
    
    def _run_indexing(self):
        """Run the full indexing pipeline."""
        from src.ingestion.indexer import run_full_indexing_pipeline
        run_full_indexing_pipeline(settings)
    
    def run_query(self, query: str, mode: str = "suggest", category: str = None):
        """Execute a full RAG query."""
        self.query_history.append(query)
        tracer = PipelineTracer()
        trace = tracer.start_trace(query)
        
        # Step 1: Query Transformation (HyDE)
        step = tracer.begin_step(StepType.QUERY_TRANSFORM, "Query Transformation (HyDE)")
        hyde_query = query  
        try:
            hyde_doc = self.hyde_chain.invoke({"query": query})
            hyde_query = hyde_doc
            step = tracer.end_step(step, data={
                "original": query,
                "hyde_preview": hyde_doc[:100] + "..."
            })
        except Exception as e:
            step = tracer.end_step(step, success=False, error=str(e))
        render_step(step)
        
        # Step 2: Dense Retrieval
        step = tracer.begin_step(StepType.DENSE_RETRIEVAL, "Dense Retrieval")
        dense_results = self.retriever.dense_search(hyde_query, k=settings.retrieval_k * 2)
        step = tracer.end_step(step, data={
            "count": len(dense_results),
            "top_results": [{"file": d.metadata.get("source", "?"), "score": s} for d, s in dense_results[:3]]
        })
        render_step(step)
        
        # Step 3: BM25 Retrieval
        step = tracer.begin_step(StepType.BM25_RETRIEVAL, "Sparse Retrieval")
        bm25_results = self.retriever.bm25_search(query, k=settings.retrieval_k * 2)
        step = tracer.end_step(step, data={
            "count": len(bm25_results),
            "top_results": [{"file": d.metadata.get("source", "?"), "score": min(s / 20, 1.0)} for d, s in bm25_results[:3]]
        })
        render_step(step)
        
        # Step 4: RRF Fusion
        step = tracer.begin_step(StepType.RRF_FUSION, "Rank Fusion")
        fused = self.retriever.reciprocal_rank_fusion(dense_results, bm25_results)
        overlap = len(set(d.metadata["source"] for d, _ in dense_results[:settings.retrieval_k]) & 
                     set(d.metadata["source"] for d, _ in bm25_results[:settings.retrieval_k]))
        step = tracer.end_step(step, data={"total_candidates": len(fused), "overlap": overlap})
        render_step(step)
        
        # Step 5: Reranking
        step = tracer.begin_step(StepType.RERANKING, "Cross-Encoder Reranking")
        retrieved_chunks = self.retriever.rerank(query, fused)
        
        if retrieved_chunks:
            step = tracer.end_step(step, data={
                "results": [{"rank": c.final_rank, "file": c.document.metadata.get("source", "?"), "score": c.rerank_score} for c in retrieved_chunks],
                "top_score": retrieved_chunks[0].rerank_score if retrieved_chunks else 0
            })
        else:
            step = tracer.end_step(step, success=False, error="No results after reranking")
        render_step(step)
        
        # Pre-generation Gate
        should_generate, refusal_reason = pre_retrieval_gate(
            retrieved_chunks, 
            min_rerank_score=settings.low_confidence
        )
        
        if not should_generate:
            tracer.finish_trace()
            render_pipeline_summary(trace)
            console.print(Panel(
                refusal_reason,
                title="[dim]╭─[/dim] [bold yellow]INSUFFICIENT CONTEXT[/bold yellow] [dim]─[/dim]",
                border_style="dim"
            ))
            return
        
        # Step 6: Confidence Pre-Check
        step = tracer.begin_step(StepType.CONFIDENCE_CHECK, "Confidence Check")
        _scores  = [c.rerank_score for c in retrieved_chunks]
        _weights = [1.0 / (i + 1) for i in range(len(_scores))]
        retrieval_score = min(sum(s * w for s, w in zip(_scores, _weights)) / sum(_weights), 1.0)
        level = "HIGH" if retrieval_score >= settings.high_confidence else "MEDIUM" if retrieval_score >= settings.medium_confidence else "LOW"
        step = tracer.end_step(step, data={"confidence": retrieval_score, "level": level})
        render_step(step)
        
        # Step 7: Generation
        step = tracer.begin_step(StepType.GENERATION, "Generation")
        if mode == "explain":
            result_text, gen_provider = self.generator.explain_algorithm(query, retrieved_chunks)
            result = {"explanation": result_text, "algorithms": []}
        else:
            result, gen_provider = self.generator.suggest_algorithms(query, retrieved_chunks)

        step.description = f"Generation ({gen_provider})"
        answer_len = sum(len(str(v)) for v in result.values())
        step = tracer.end_step(step, data={
            "model": gen_provider,
            "tokens": answer_len // 4,
            "chunks_used": len(retrieved_chunks)
        })
        render_step(step)
        
        # Post-generation Confidence
        confidence_report = self.confidence_scorer.assess(retrieved_chunks, result, query)
        result["_confidence"] = confidence_report
        
        # Finish Trace & Display
        tracer.finish_trace(final_answer=result)
        render_pipeline_summary(trace)
        
        # Unified rendering for all modes
        render_algorithm_results(result, trace)
        
        # Display confidence warnings
        if confidence_report.warnings:
            console.print()
            for w in confidence_report.warnings:
                console.print(f"  [dim yellow]⚠️  {w}[/dim yellow]")
    
    def _print_status_bar(self):
        """Prints the persistent 1-line status bar."""
        console.print()
        console.print(f"  [dim]FAISS ● BM25 ● {self.total_chunks:,} chunks │ /help │ /stats │ /history │ v1.0[/dim]")
        console.print()
        
    def _prompt_query(self) -> str:
        console.print("[dim]╭─[/dim] [bold cyan]QUERY[/bold cyan] [dim]──────────────────────────────────────────────────────────────[/dim]")
        try:
            val = console.input("[dim]│[/dim] [bold cyan]❯[/bold cyan] ")
            console.print("[dim]╰──────────────────────────────────────────────────────────────────────[/dim]")
            return val.strip()
        except KeyboardInterrupt:
            console.print("[dim]╰──────────────────────────────────────────────────────────────────────[/dim]")
            raise
    
    def run(self):
        """Main REPL loop."""
        self.print_banner()
        self.initialize()
        
        console.print()
        
        current_category = None
        
        while True:
            try:
                raw_input = self._prompt_query()
                
                try:
                    import sys, termios
                    if sys.stdin.isatty():
                        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
                except Exception:
                    pass  

                if not raw_input:
                    continue
                
                # Command handling
                if raw_input.lower() in ("/quit", "/exit", "quit", "exit"):
                    console.print("\n  [dim]Session terminated.[/dim]\n")
                    break
                
                elif raw_input.lower() == "/help":
                    console.print()
                    console.print(Panel(
                        HELP_TEXT, 
                        title="[dim]╭─[/dim] [bold]COMMANDS & TIPS[/bold] [dim]─[/dim]", 
                        title_align="left", 
                        border_style="dim cyan", 
                        box=box.ROUNDED,
                        padding=(1, 2)
                    ))
                
                elif raw_input.lower() == "/stats":
                    indexer = AlgoIndexer(settings.cohere_api_key, settings.faiss_db_path)
                    stats = indexer.get_stats()
                    table = Table(box=None, show_header=False, padding=(0, 2))
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="dim white")
                    for k, v in stats.items():
                        table.add_row(str(k), str(v))
                    console.print()
                    console.print(Panel(table, title="[dim]╭─[/dim] [bold]STATISTICS[/bold] [dim]─[/dim]", title_align="left", border_style="dim", box=box.ROUNDED))
                
                elif raw_input.lower() == "/history":
                    console.print()
                    if self.query_history:
                        for i, q in enumerate(self.query_history[-10:], 1):
                            console.print(f"  [dim][{i:02d}][/dim] {q}")
                    else:
                        console.print("  [dim]No history.[/dim]")
                    console.print()
                
                elif raw_input.lower() == "/index":
                    console.print("\n  [yellow]Re-indexing...[/yellow]")
                    self._run_indexing()
                
                elif raw_input.lower().startswith("/category "):
                    current_category = raw_input[10:].strip()
                    console.print(f"\n  [green]Category filter set: {current_category}[/green]")
                
                elif raw_input.lower().startswith("/explain "):
                    query = raw_input[9:].strip()
                    self.run_query(query, mode="explain", category=current_category)
                
                else:
                    self.run_query(raw_input, mode="suggest", category=current_category)
                    
                self._print_status_bar()
            
            except KeyboardInterrupt:
                console.print("\n  [dim]Use /quit to exit[/dim]\n")
            except EOFError:
                break
            except Exception as e:
                console.print(f"\n  [red]Error: {e}[/red]")
                logger.exception("Query failed")

def main():
    app = AlgoRAGApp()
    app.run()

if __name__ == "__main__":
    main()