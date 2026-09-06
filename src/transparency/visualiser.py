"""
TRACE Visualiser - Minimal, technical, elegant UI rendering.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.markdown import Markdown as RichMarkdown

from .tracer import TraceStep, StepType, RAGTrace

console = Console()

# Semantic Colors
C_PRIMARY = "cyan"
C_SUCCESS = "green"
C_CMD = "yellow"
C_ERROR = "red"
C_SEC = "dim white"
C_NORM = "white"

C_BM25 = "blue"
C_VECTOR = "magenta"
C_FUSION = "magenta"
C_RERANK = "yellow"
C_GEN = "cyan"

def render_query_banner(query: str):
    """Query banner is now rendered in app.py directly via the prompt."""
    pass

def render_step(step: TraceStep):
    """
    In TRACE, we remain silent during the pipeline steps to keep the terminal
    clean. The pipeline summary will display the full trace once complete.
    """
    pass

def render_pipeline_summary(trace: RAGTrace):
    """Render the ╭─ RETRIEVAL TRACE ─ box."""
    bm25_count = 0
    dense_count = 0
    fusion_count = 0
    rerank_count = 0
    
    for s in trace.steps:
        if s.step_type == StepType.BM25_RETRIEVAL:
            bm25_count = s.data.get("count", 0)
        elif s.step_type == StepType.DENSE_RETRIEVAL:
            dense_count = s.data.get("count", 0)
        elif s.step_type == StepType.RRF_FUSION:
            fusion_count = s.data.get("total_candidates", 0)
        elif s.step_type == StepType.RERANKING:
            results = s.data.get("results", [])
            rerank_count = len(results)

    trace_text = Text()
    trace_text.append("\n  [bold]🔍 Query[/bold]\n", style=C_NORM)
    trace_text.append("      │\n", style="dim")
    trace_text.append("      ├── ", style="dim")
    trace_text.append(f"📚 BM25 Search          {bm25_count} candidates\n", style=C_BM25)
    trace_text.append("      ├── ", style="dim")
    trace_text.append(f"⚡ Vector Search        {dense_count} candidates\n", style=C_VECTOR)
    trace_text.append("      │\n", style="dim")
    trace_text.append("      └── ", style="dim")
    trace_text.append(f"🔀 Rank Fusion          {fusion_count} candidates\n", style=C_FUSION)
    trace_text.append("               │\n", style="dim")
    trace_text.append("               └── ", style="dim")
    trace_text.append(f"🎯 Top-K          {rerank_count} chunks\n\n", style=C_RERANK)

    console.print()
    console.print(Panel(
        trace_text,
        title=f"[dim]╭─[/dim] [bold {C_NORM}]RETRIEVAL TRACE[/bold {C_NORM}] [dim]─[/dim]",
        title_align="left",
        border_style="dim cyan",
        box=box.ROUNDED,
        padding=(1, 3)
    ))

def render_algorithm_results(result: dict, trace: RAGTrace = None):
    """Render the generated answer and the sources section."""
    has_algorithms = bool(result.get("algorithms"))
    has_explanation = bool(result.get("explanation"))
    
    if not has_algorithms and not has_explanation:
        console.print(Panel(
            "No suitable algorithms found in the knowledge base for this query.",
            title="[dim]╭─[/dim] [bold yellow]NO RESULTS[/bold yellow] [dim]─[/dim]",
            border_style="dim"
        ))
        return
    
    # ── ANSWER ──
    md_content = ""
    if has_explanation:
        md_content += f"{result.get('explanation')}\n\n"
        
    if result.get('understanding'):
        md_content += f"*{result.get('understanding')}*\n\n"

    for algo in result.get("algorithms", []):
        md_content += f"**{algo.get('name', 'Algorithm')}**\n"
        md_content += f"Complexity: Time `O({algo.get('time_complexity', '?')})` | Space `O({algo.get('space_complexity', '?')})`\n"
        md_content += f"{algo.get('reason', '')}\n\n"

    if result.get("comparison"):
        md_content += f"**Comparison:**\n{result.get('comparison')}\n\n"
    if result.get("caveats"):
        md_content += f"**Caveats:**\n{result.get('caveats')}\n"

    console.print()
    console.print(Panel(
        RichMarkdown(md_content.strip()),
        title=f"[dim]╭─[/dim] [bold {C_SUCCESS}]ANSWER[/bold {C_SUCCESS}] [dim]─[/dim]",
        title_align="left",
        border_style=C_SUCCESS,
        box=box.ROUNDED,
        padding=(1, 2)
    ))

    # ── SOURCES ──
    console.print("\n  [bold dim]SOURCES[/bold dim]")
    console.print("  [dim]────────────────────────────────────────────────────────────────────[/dim]")
    
    sources = []
    if trace:
        for s in trace.steps:
            if s.step_type == StepType.RERANKING:
                sources = s.data.get("results", [])
                break
    
    if sources:
        for i, s in enumerate(sources, 1):
            file_path = s.get("file", "unknown")
            score = s.get("score", 0.0)
            console.print(f"  [dim][{i:02d}][/dim] [{C_PRIMARY}]{file_path}[/{C_PRIMARY}]")
            console.print(f"       [dim]relevance  {score:.2f}[/dim]\n")
    else:
        for i, algo in enumerate(result.get("algorithms", []), 1):
            file_path = algo.get("source_file", "unknown")
            score = algo.get("confidence", 0.0)
            console.print(f"  [dim][{i:02d}][/dim] [{C_PRIMARY}]{file_path}[/{C_PRIMARY}]")
            console.print(f"       [dim]relevance  {score:.2f}[/dim]\n")