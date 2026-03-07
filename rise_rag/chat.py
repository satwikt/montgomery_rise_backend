"""
chat.py — RISE RAG Chatbot CLI entry point.

Run from the project root:
    python chat.py

Commands during chat:
    /sources    — Show sources for the last answer
    /clear      — Clear conversation history
    /models     — List available Ollama models
    /status     — Show system status
    /help       — Show help
    /quit       — Exit
    /q          — Exit (shorthand)
"""

import sys
import logging
from pathlib import Path

# ─── Logging (keep quiet unless debugging) ───────────────────────────────────
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("chat")

# ─── Dependency check ─────────────────────────────────────────────────────────
def _check_dependencies() -> bool:
    missing = []
    try:
        import chromadb
    except ImportError:
        missing.append("chromadb")
    try:
        import sentence_transformers
    except ImportError:
        missing.append("sentence-transformers")
    try:
        import requests
    except ImportError:
        missing.append("requests")

    # rich is optional — we degrade gracefully
    if missing:
        print(" Missing required packages. Run:")
        print(f"   pip install {' '.join(missing)}")
        return False
    return True


if not _check_dependencies():
    sys.exit(1)

# ─── Imports ──────────────────────────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.rule import Rule
    from rich.markdown import Markdown
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from app.chatbot import RiseChatbot
from app.llm import is_ollama_running, get_available_models
from app.config import OLLAMA_MODEL, CHROMA_PERSIST_DIR

# ─── UI Helpers ───────────────────────────────────────────────────────────────

console = Console() if RICH_AVAILABLE else None


def print_banner() -> None:
    if RICH_AVAILABLE:
        console.print()
        console.print(Panel(
            "[bold cyan]RISE[/bold cyan] — Revitalization Intelligence and Smart Empowerment\n"
            "[dim]Montgomery, Alabama — Vacant Parcel Decision Support System[/dim]\n\n"
            "Type your question, or [bold]/help[/bold] for commands.",
            title="[bold green]  RISE RAG Chatbot[/bold green]",
            border_style="cyan",
        ))
        console.print()
    else:
        print("\n" + "=" * 60)
        print("  RISE RAG Chatbot")
        print("  Revitalization Intelligence and Smart Empowerment")
        print("  Montgomery, Alabama")
        print("=" * 60)
        print("Type your question, or /help for commands.\n")


def print_status(bot: RiseChatbot) -> None:
    ollama_ok = is_ollama_running()
    kb_size = bot.knowledge_base_size()

    if RICH_AVAILABLE:
        ollama_status = "[green] Running[/green]" if ollama_ok else "[red] Not running[/red]"
        kb_status = f"[green]{kb_size} chunks[/green]" if kb_size > 0 else "[red]Empty — run: python scripts/ingest.py[/red]"
        
        console.print(Panel(
            f"[bold]Ollama:[/bold]     {ollama_status} (model: {OLLAMA_MODEL})\n"
            f"[bold]Knowledge base:[/bold] {kb_status}\n"
            f"[bold]ChromaDB:[/bold]   {CHROMA_PERSIST_DIR}",
            title="System Status",
            border_style="blue",
        ))
    else:
        print(f"\nOllama: {'Running' if ollama_ok else 'Not running'} (model: {OLLAMA_MODEL})")
        print(f"Knowledge base: {kb_size} chunks")
        if kb_size == 0:
            print("    Run: python scripts/ingest.py")
        print()


def print_help() -> None:
    help_text = (
        "\n[bold cyan]Available commands:[/bold cyan]\n"
        "  [bold]/sources[/bold]   Show the source chunks used for the last answer\n"
        "  [bold]/clear[/bold]     Clear conversation history\n"
        "  [bold]/models[/bold]    List available Ollama models\n"
        "  [bold]/status[/bold]    Show system status (Ollama, knowledge base)\n"
        "  [bold]/help[/bold]      Show this help\n"
        "  [bold]/quit[/bold]      Exit the chatbot\n\n"
        "[bold cyan]Parcel filters (prefix your question):[/bold cyan]\n"
        "  [bold]@A[/bold] your question   — Restrict retrieval to Parcel A (Heritage)\n"
        "  [bold]@B[/bold] your question   — Restrict retrieval to Parcel B (IX Hub)\n"
        "  [bold]@C[/bold] your question   — Restrict retrieval to Parcel C (Food Desert)\n\n"
        "[bold cyan]Example questions:[/bold cyan]\n"
        "  What is RISE?\n"
        "  What is the heritage boost and why does it matter?\n"
        "  @C What grants are available for Parcel C?\n"
        "  How much would the city pay for a grocery store?\n"
        "  What is the most urgent grant right now?\n"
        "  How does the 311 distress score work?\n"
    )
    if RICH_AVAILABLE:
        console.print(Panel(help_text, title="Help", border_style="yellow"))
    else:
        # Strip rich markup for plain output
        import re
        plain = re.sub(r'\[.*?\]', '', help_text)
        print(plain)


def print_answer(answer: str, question: str) -> None:
    if RICH_AVAILABLE:
        console.print()
        console.print(Rule("[bold cyan]RISE[/bold cyan]", style="cyan"))
        # Try to render as markdown (Ollama often returns markdown)
        try:
            console.print(Markdown(answer))
        except Exception:
            console.print(answer)
        console.print()
    else:
        print(f"\nRISE: {answer}\n")


def print_sources(sources: list[dict]) -> None:
    if not sources:
        print("No sources available.")
        return

    if RICH_AVAILABLE:
        lines = []
        for i, src in enumerate(sources, 1):
            title = src.get("document_title", "Unknown")
            parcel = src.get("parcel_id", "?")
            topic = src.get("topic", "?")
            file_ = src.get("source_file", "?")
            lines.append(f"  [dim]{i}.[/dim] [bold]{title}[/bold]")
            lines.append(f"     parcel=[cyan]{parcel}[/cyan] | topic=[cyan]{topic}[/cyan] | file=[dim]{file_}[/dim]")
        console.print(Panel(
            "\n".join(lines),
            title=f"Sources ({len(sources)} chunks retrieved)",
            border_style="dim",
        ))
    else:
        print(f"\nSources ({len(sources)} chunks retrieved):")
        for i, src in enumerate(sources, 1):
            print(f"  {i}. {src.get('document_title', 'Unknown')}")
            print(f"     parcel={src.get('parcel_id')} | topic={src.get('topic')}")
        print()


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main() -> None:
    print_banner()

    # ── Initialise chatbot ────────────────────────────────────────────────────
    if RICH_AVAILABLE:
        with console.status("[cyan]Initialising RISE chatbot...[/cyan]"):
            bot = RiseChatbot()
    else:
        print("Initialising RISE chatbot...")
        bot = RiseChatbot()

    # ── Check knowledge base ──────────────────────────────────────────────────
    if not bot.is_ready():
        if RICH_AVAILABLE:
            console.print(
                "[bold red]  Knowledge base is empty![/bold red]\n"
                "Run [bold]python scripts/ingest.py[/bold] first to load the RISE documents.",
                style="red",
            )
        else:
            print("  Knowledge base is empty!")
            print("Run: python scripts/ingest.py\n")
        sys.exit(1)

    # ── Check Ollama ──────────────────────────────────────────────────────────
    if not is_ollama_running():
        if RICH_AVAILABLE:
            console.print(Panel(
                "[yellow]Ollama is not running. Answers will show raw retrieved context instead of AI-generated responses.[/yellow]\n\n"
                "To enable AI answers:\n"
                "  1. Install Ollama: [link]https://ollama.com/download[/link]\n"
                f"  2. Pull a model:  [bold]ollama pull {OLLAMA_MODEL}[/bold]\n"
                "  3. Start server:  [bold]ollama serve[/bold]",
                title="  Ollama not detected",
                border_style="yellow",
            ))
        else:
            print("  Ollama not running. Raw context mode active.")
            print(f"   Install Ollama and run: ollama pull {OLLAMA_MODEL}\n")
    else:
        models = get_available_models()
        if OLLAMA_MODEL not in models and not any(OLLAMA_MODEL in m for m in models):
            model_list = ", ".join(models) if models else "none found"
            if RICH_AVAILABLE:
                console.print(
                    f"[yellow]  Model '{OLLAMA_MODEL}' not found in Ollama.[/yellow]\n"
                    f"   Available: {model_list}\n"
                    f"   Run: [bold]ollama pull {OLLAMA_MODEL}[/bold]",
                )
            else:
                print(f"  Model '{OLLAMA_MODEL}' not found. Available: {model_list}")
                print(f"   Run: ollama pull {OLLAMA_MODEL}\n")

    if RICH_AVAILABLE:
        console.print(
            f"[green] Ready![/green] Knowledge base: [bold]{bot.knowledge_base_size()} chunks[/bold]. "
            f"Type [bold]/help[/bold] for commands.\n"
        )
    else:
        print(f" Ready! {bot.knowledge_base_size()} chunks loaded. Type /help for commands.\n")

    # ── Keep last response for /sources command ───────────────────────────────
    last_sources: list[dict] = []

    # ── Chat loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            if RICH_AVAILABLE:
                user_input = console.input("[bold green]You:[/bold green] ").strip()
            else:
                user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        cmd = user_input.lower()

        if cmd in {"/quit", "/q", "/exit", "quit", "exit"}:
            print("\nGoodbye!\n")
            sys.exit(0)

        elif cmd == "/help":
            print_help()
            continue

        elif cmd == "/status":
            print_status(bot)
            continue

        elif cmd == "/clear":
            bot.clear_history()
            print(" Conversation history cleared.\n" if not RICH_AVAILABLE
                  else "")
            if RICH_AVAILABLE:
                console.print("[green] Conversation history cleared.[/green]\n")
            continue

        elif cmd == "/sources":
            print_sources(last_sources)
            continue

        elif cmd == "/models":
            models = get_available_models()
            if models:
                model_str = "\n".join(f"  - {m}" for m in models)
                if RICH_AVAILABLE:
                    console.print(f"[bold]Available Ollama models:[/bold]\n{model_str}\n")
                else:
                    print(f"Available Ollama models:\n{model_str}\n")
            else:
                msg = "No Ollama models found (is Ollama running?)"
                print(msg)
            continue

        # ── Parcel filter (@A, @B, @C prefix) ─────────────────────────────────
        parcel_filter: str | None = None
        question = user_input

        if user_input.upper().startswith("@A "):
            parcel_filter = "A"
            question = user_input[3:].strip()
        elif user_input.upper().startswith("@B "):
            parcel_filter = "B"
            question = user_input[3:].strip()
        elif user_input.upper().startswith("@C "):
            parcel_filter = "C"
            question = user_input[3:].strip()

        if not question:
            continue

        # ── Generate answer ───────────────────────────────────────────────────
        if RICH_AVAILABLE:
            with console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
                response = bot.ask(question, parcel_filter=parcel_filter)
        else:
            print("Thinking...")
            response = bot.ask(question, parcel_filter=parcel_filter)

        last_sources = response.sources
        print_answer(response.answer, question)

        # Show source count hint
        if RICH_AVAILABLE and response.num_chunks_retrieved > 0:
            console.print(
                f"[dim]Retrieved {response.num_chunks_retrieved} source chunks. "
                "Type [bold]/sources[/bold] to see them.[/dim]\n"
            )


if __name__ == "__main__":
    main()
