"""
chat.py — RISE RAG Chatbot CLI entry point.

Run from the project root:
    python chat.py

Commands during chat:
    /sources    — Show sources for the last answer
    /clear      — Clear conversation history
    /status     — Show system status
    /help       — Show help
    /quit       — Exit

Parcel filters (prefix your question):
    @A question   — Restrict retrieval to Parcel A (Heritage)
    @B question   — Restrict retrieval to Parcel B (IX Hub)
    @C question   — Restrict retrieval to Parcel C (Food Desert)
"""

import sys
import logging

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("chat")


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
        import groq
    except ImportError:
        missing.append("groq")
    if missing:
        print("Missing required packages. Run:")
        print(f"   pip install {' '.join(missing)}")
        return False
    return True


if not _check_dependencies():
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from app.chatbot import RiseChatbot
from app.llm import is_llm_available, get_llm_info
from app.config import GROQ_MODEL, CHROMA_PERSIST_DIR

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
        print("  RISE RAG Chatbot — Montgomery, Alabama")
        print("=" * 60)
        print("Type your question, or /help for commands.\n")


def print_status(bot: RiseChatbot) -> None:
    llm_info = get_llm_info()
    kb_size = bot.knowledge_base_size()
    if RICH_AVAILABLE:
        llm_status = (
            "[green]Configured[/green]" if llm_info["configured"]
            else "[red]Not configured — add GROQ_API_KEY to .env[/red]"
        )
        kb_status = (
            f"[green]{kb_size} chunks[/green]" if kb_size > 0
            else "[red]Empty — run: python scripts/ingest.py[/red]"
        )
        console.print(Panel(
            f"[bold]LLM provider:[/bold]   {llm_info['provider']} ({llm_info['model']})\n"
            f"[bold]LLM status:[/bold]     {llm_status}\n"
            f"[bold]Hosting:[/bold]        {llm_info['hosting']}\n"
            f"[bold]Knowledge base:[/bold] {kb_status}\n"
            f"[bold]ChromaDB:[/bold]       {CHROMA_PERSIST_DIR}",
            title="System Status", border_style="blue",
        ))
    else:
        print(f"\nLLM: {llm_info['provider']} ({llm_info['model']})")
        print(f"Configured: {llm_info['configured']}")
        print(f"Knowledge base: {kb_size} chunks\n")


def print_help() -> None:
    help_text = (
        "\n[bold cyan]Commands:[/bold cyan]\n"
        "  [bold]/sources[/bold]   Show source chunks for the last answer\n"
        "  [bold]/clear[/bold]     Clear conversation history\n"
        "  [bold]/status[/bold]    Show system status\n"
        "  [bold]/help[/bold]      Show this help\n"
        "  [bold]/quit[/bold]      Exit\n\n"
        "[bold cyan]Parcel filters:[/bold cyan]\n"
        "  [bold]@A[/bold] question  — Parcel A (Heritage)\n"
        "  [bold]@B[/bold] question  — Parcel B (IX Hub)\n"
        "  [bold]@C[/bold] question  — Parcel C (Food Desert)\n\n"
        "[bold cyan]Examples:[/bold cyan]\n"
        "  What is RISE?\n"
        "  @C What grants are available for Parcel C?\n"
        "  What is the most urgent grant right now?\n"
        "  How does the 311 distress score work?\n"
    )
    if RICH_AVAILABLE:
        console.print(Panel(help_text, title="Help", border_style="yellow"))
    else:
        import re
        print(re.sub(r'\[.*?\]', '', help_text))


def print_answer(answer: str) -> None:
    if RICH_AVAILABLE:
        console.print()
        console.print(Rule("[bold cyan]RISE[/bold cyan]", style="cyan"))
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
            lines.append(f"  [dim]{i}.[/dim] [bold]{src.get('document_title', 'Unknown')}[/bold]")
            lines.append(
                f"     parcel=[cyan]{src.get('parcel_id', '?')}[/cyan] | "
                f"topic=[cyan]{src.get('topic', '?')}[/cyan] | "
                f"file=[dim]{src.get('source_file', '?')}[/dim]"
            )
        console.print(Panel("\n".join(lines), title=f"Sources ({len(sources)} chunks)", border_style="dim"))
    else:
        print(f"\nSources ({len(sources)} chunks):")
        for i, src in enumerate(sources, 1):
            print(f"  {i}. {src.get('document_title', 'Unknown')} | parcel={src.get('parcel_id')}")
        print()


def main() -> None:
    print_banner()

    if RICH_AVAILABLE:
        with console.status("[cyan]Initialising RISE chatbot...[/cyan]"):
            bot = RiseChatbot()
    else:
        print("Initialising RISE chatbot...")
        bot = RiseChatbot()

    if not bot.is_ready():
        print("Knowledge base is empty! Run: python scripts/ingest.py")
        sys.exit(1)

    if not is_llm_available():
        if RICH_AVAILABLE:
            console.print(Panel(
                "[yellow]GROQ_API_KEY is not set.[/yellow]\n"
                "Answers will show raw context only.\n\n"
                "To enable AI answers:\n"
                "  1. Get a free key at [link]https://console.groq.com[/link]\n"
                "  2. Add to [bold].env[/bold]: GROQ_API_KEY=your_key_here",
                title="  Groq not configured", border_style="yellow",
            ))
        else:
            print("GROQ_API_KEY not set. Get a free key at https://console.groq.com\n")

    if RICH_AVAILABLE:
        llm_info = get_llm_info()
        console.print(
            f"[green]Ready![/green] "
            f"[bold]{bot.knowledge_base_size()} chunks[/bold] loaded | "
            f"LLM: [bold]{llm_info['provider']} ({llm_info['model']})[/bold]. "
            f"Type [bold]/help[/bold] for commands.\n"
        )
    else:
        print(f"Ready! {bot.knowledge_base_size()} chunks loaded. Type /help for commands.\n")

    last_sources: list[dict] = []

    while True:
        try:
            user_input = (
                console.input("[bold green]You:[/bold green] ").strip()
                if RICH_AVAILABLE else input("You: ").strip()
            )
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue

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
            if RICH_AVAILABLE:
                console.print("[green]Conversation history cleared.[/green]\n")
            else:
                print("Conversation history cleared.\n")
            continue
        elif cmd == "/sources":
            print_sources(last_sources)
            continue

        parcel_filter: str | None = None
        question = user_input
        if user_input.upper().startswith("@A "):
            parcel_filter, question = "A", user_input[3:].strip()
        elif user_input.upper().startswith("@B "):
            parcel_filter, question = "B", user_input[3:].strip()
        elif user_input.upper().startswith("@C "):
            parcel_filter, question = "C", user_input[3:].strip()

        if not question:
            continue

        if RICH_AVAILABLE:
            with console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
                response = bot.ask(question, parcel_filter=parcel_filter)
        else:
            print("Thinking...")
            response = bot.ask(question, parcel_filter=parcel_filter)

        last_sources = response.sources
        print_answer(response.answer)

        if RICH_AVAILABLE and response.num_chunks_retrieved > 0:
            console.print(
                f"[dim]Retrieved {response.num_chunks_retrieved} chunks. "
                "Type [bold]/sources[/bold] to see them.[/dim]\n"
            )


if __name__ == "__main__":
    main()
