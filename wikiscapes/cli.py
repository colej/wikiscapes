"""Wikiscapes CLI — Typer application.

Commands:
  init          Initialize a new wiki project in the current directory
  ingest        Ingest a raw source file into the wiki
  query         Query the wiki using topographic routing
  evolve        Re-layout the topographic map and apply plasticity
  lint          Run health checks
  map           Render the topographic map
  status        Print wiki statistics
  add-raw       Copy a file into raw/ with validation
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="wikiscapes",
    help="Brainscapes-inspired topographic LLM knowledge base wiki.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


def _load_config() -> object:
    from wikiscapes.config import load_config
    return load_config()


def _load_client(config: object) -> object:
    from wikiscapes.llm.client import WikiClient
    return WikiClient.from_config(config)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Directory to initialize (default: current)")] = Path("."),
) -> None:
    """Initialize a new wikiscapes wiki project."""
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    (path / "wiki").mkdir(exist_ok=True)
    (path / "raw").mkdir(exist_ok=True)
    (path / ".wikiscapes").mkdir(exist_ok=True)

    # wikiscapes.toml
    toml_path = path / "wikiscapes.toml"
    if not toml_path.exists():
        from wikiscapes.config import DEFAULT_TOML
        toml_path.write_text(DEFAULT_TOML, encoding="utf-8")
        console.print(f"[green]✓[/green] Created wikiscapes.toml")
    else:
        console.print("[yellow]⚠[/yellow] wikiscapes.toml already exists — skipping")

    # CLAUDE.md schema governance
    claude_md = path / "CLAUDE.md"
    if not claude_md.exists():
        claude_md.write_text(_CLAUDE_MD_TEMPLATE, encoding="utf-8")
        console.print(f"[green]✓[/green] Created CLAUDE.md")

    # wiki/index.md
    index_path = path / "wiki" / "index.md"
    if not index_path.exists():
        from wikiscapes.store.index import _HEADER
        index_path.write_text(_HEADER, encoding="utf-8")
        console.print(f"[green]✓[/green] Created wiki/index.md")

    # .gitignore entry for .wikiscapes/
    gitignore = path / ".gitignore"
    gitignore_entry = ".wikiscapes/\n"
    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8")
        if ".wikiscapes" not in content:
            gitignore.write_text(content + gitignore_entry, encoding="utf-8")
    else:
        gitignore.write_text(gitignore_entry, encoding="utf-8")
    console.print(f"[green]✓[/green] Updated .gitignore")

    console.print()
    console.print(Panel(
        "[bold]Wiki initialized![/bold]\n\n"
        "Next steps:\n"
        "  1. Set [cyan]ANTHROPIC_API_KEY[/cyan] in your environment\n"
        "  2. Drop source files into [cyan]raw/[/cyan]\n"
        "  3. [cyan]wikiscapes ingest raw/yourfile.txt[/cyan]\n"
        "  4. After 20+ articles: [cyan]wikiscapes evolve[/cyan]\n"
        "  5. [cyan]wikiscapes query \"your question here\"[/cyan]",
        title="wikiscapes init",
    ))


# ---------------------------------------------------------------------------
# add-raw
# ---------------------------------------------------------------------------

@app.command("add-raw")
def add_raw(
    source: Annotated[Path, typer.Argument(help="File to add to raw/")],
) -> None:
    """Copy a file into raw/ with basic validation."""
    config = _load_config()
    raw_dir = config.raw_dir_path()  # type: ignore
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        err_console.print(f"[red]Error:[/red] File not found: {source}")
        raise typer.Exit(1)

    dest = raw_dir / source.name
    if dest.exists():
        err_console.print(f"[yellow]⚠[/yellow] {dest.name} already exists in raw/ — skipping")
        raise typer.Exit(0)

    shutil.copy2(str(source), str(dest))
    console.print(f"[green]✓[/green] Added {source.name} to raw/")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@app.command()
def ingest(
    source: Annotated[Path, typer.Argument(help="Path to source file")],
    article_id: Annotated[Optional[str], typer.Option("--id", help="Override article ID slug")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview without writing")] = False,
) -> None:
    """Ingest a raw source file into the wiki."""
    if not source.exists():
        err_console.print(f"[red]Error:[/red] Source file not found: {source}")
        raise typer.Exit(1)

    config = _load_config()
    client = _load_client(config)

    with console.status(f"[bold]Ingesting {source.name}…"):
        from wikiscapes.core.ingest import ingest as _ingest
        article = _ingest(
            source_path=source,
            config=config,  # type: ignore
            client=client,  # type: ignore
            article_id=article_id,
            dry_run=dry_run,
        )

    fm = article.frontmatter
    action = "[dim](dry run — not saved)[/dim]" if dry_run else f"→ wiki/{fm.id}.md"
    console.print(f"[green]✓[/green] [bold]{fm.title}[/bold] [{fm.id}] {action}")

    if dry_run:
        console.print()
        console.print(article.body[:800] + ("…" if len(article.body) > 800 else ""))

    _maybe_suggest_evolve(config)  # type: ignore


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to ask the wiki")],
    k: Annotated[int, typer.Option("--k", help="Articles to retrieve")] = 6,
    no_map: Annotated[bool, typer.Option("--no-map", help="Disable topographic routing")] = False,
    show_sources: Annotated[bool, typer.Option("--show-sources")] = False,
) -> None:
    """Query the wiki using topographic routing."""
    config = _load_config()
    client = _load_client(config)

    with console.status("[bold]Routing query through topographic map…"):
        from wikiscapes.core.query import query as _query
        result = _query(
            question=question,
            config=config,  # type: ignore
            client=client,  # type: ignore
            k_results=k,
            use_map=not no_map,
        )

    if result.cluster_hint:
        console.print(f"[dim]Map region: {result.cluster_hint}[/dim]")

    console.print()
    console.print(result.answer)

    if result.gap_detected:
        console.print()
        console.print("[yellow]⚠ Knowledge gap detected — consider ingesting more sources on this topic.[/yellow]")

    if show_sources and result.sources:
        console.print()
        console.print(f"[dim]Sources: {', '.join(result.sources)}[/dim]")


# ---------------------------------------------------------------------------
# evolve
# ---------------------------------------------------------------------------

@app.command()
def evolve(
    force: Annotated[bool, typer.Option("--force", help="Run even if thresholds not met")] = False,
    no_bridges: Annotated[bool, typer.Option("--no-bridges", help="Skip bridge article generation")] = False,
) -> None:
    """Re-layout the topographic map and apply plasticity updates."""
    config = _load_config()

    # Check thresholds
    if not force:
        from wikiscapes.core.evolve import should_suggest_evolve
        should, reason = should_suggest_evolve(config)  # type: ignore
        if not should:
            console.print("[yellow]ℹ[/yellow] Evolution thresholds not yet crossed.")
            console.print("  Use [cyan]--force[/cyan] to run anyway.")
            return
        console.print(f"[dim]{reason}[/dim]")

    client = _load_client(config)

    with console.status("[bold]Evolving topographic map…"):
        from wikiscapes.core.evolve import evolve as _evolve
        result = _evolve(
            config=config,  # type: ignore
            client=client,  # type: ignore
            force=force,
            generate_bridges=not no_bridges,
        )

    console.print(f"[green]✓[/green] Evolution complete (layout v{result.layout_version})")
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_row("Articles re-embedded:", str(result.articles_re_embedded))
    t.add_row("Clusters changed:", str(result.clusters_changed))
    t.add_row("Bridge articles generated:", str(result.bridges_generated))
    t.add_row("Duration:", f"{result.duration_seconds:.1f}s")
    console.print(t)


# ---------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------

@app.command()
def lint(
    fix: Annotated[bool, typer.Option("--fix", help="Auto-repair fixable issues")] = False,
    deep: Annotated[bool, typer.Option("--deep", help="Include Haiku semantic checks (costs tokens)")] = False,
) -> None:
    """Run wiki health checks."""
    config = _load_config()
    client = _load_client(config) if deep else None

    with console.status("[bold]Running lint checks…"):
        from wikiscapes.core.lint import lint as _lint
        report = _lint(config=config, client=client, fix=fix, deep=deep)  # type: ignore

    # Summary
    errors = [i for i in report.issues if i.severity == "error"]
    warnings = [i for i in report.issues if i.severity == "warning"]
    infos = [i for i in report.issues if i.severity == "info"]

    status_color = "red" if errors else "yellow" if warnings else "green"
    console.print(
        f"[{status_color}]{'✗' if errors else '⚠' if warnings else '✓'}[/{status_color}] "
        f"{report.total_articles} articles | "
        f"{len(errors)} errors | {len(warnings)} warnings | {len(infos)} info"
    )
    console.print(
        f"  Topology health: {report.topology_health:.0%} | "
        f"Map coverage: {report.map_coverage:.0%} | "
        f"Stubs: {len(report.stubs)} | Orphans: {len(report.orphans)}"
    )

    for issue in report.issues:
        color = {"error": "red", "warning": "yellow", "info": "dim"}.get(issue.severity, "white")
        prefix = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(issue.severity, "·")
        article_str = f"[{issue.article_id}] " if issue.article_id else ""
        console.print(f"  [{color}]{prefix}[/{color}] {article_str}{issue.message}")


# ---------------------------------------------------------------------------
# map
# ---------------------------------------------------------------------------

@app.command(name="map")
def render_map(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("map.html"),
    fmt: Annotated[str, typer.Option("--format", "-f", help="html | png | ascii")] = "html",
    highlight: Annotated[Optional[str], typer.Option("--highlight", help="Comma-separated article IDs to highlight")] = None,
) -> None:
    """Render the topographic map."""
    config = _load_config()
    wiki_dir = config.wiki_dir_path()  # type: ignore
    state_dir = config.state_dir_path()  # type: ignore

    from wikiscapes.store.article_store import load_all_articles
    from wikiscapes.store.map_state import load_map_state

    articles = load_all_articles(wiki_dir)
    map_state = load_map_state(state_dir)

    if map_state is None or not map_state.articles:
        err_console.print("[red]Error:[/red] No map state found. Run [cyan]wikiscapes evolve[/cyan] first.")
        raise typer.Exit(1)

    highlight_ids = [h.strip() for h in highlight.split(",")] if highlight else None

    if fmt == "html":
        from wikiscapes.viz.plotly_map import render_interactive_map
        render_interactive_map(map_state, articles, output, highlight_ids=highlight_ids)
        console.print(f"[green]✓[/green] Interactive map written to {output}")

    elif fmt == "png":
        from wikiscapes.viz.static_map import render_static_map
        render_static_map(map_state, articles, output)
        console.print(f"[green]✓[/green] PNG map written to {output}")

    elif fmt == "ascii":
        from wikiscapes.viz.static_map import render_ascii_map
        text = render_ascii_map(map_state, articles)
        console.print(text)

    else:
        err_console.print(f"[red]Unknown format:[/red] {fmt}. Use html, png, or ascii.")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@app.command()
def status() -> None:
    """Print wiki statistics."""
    config = _load_config()
    wiki_dir = config.wiki_dir_path()  # type: ignore
    state_dir = config.state_dir_path()  # type: ignore

    from wikiscapes.store.article_store import load_all_articles
    from wikiscapes.store.map_state import load_map_state

    articles = load_all_articles(wiki_dir)
    map_state = load_map_state(state_dir)

    t = Table(title="Wikiscapes Status", show_header=False, box=None, padding=(0, 2))
    t.add_row("Articles:", str(len(articles)))
    t.add_row("Stubs:", str(sum(1 for a in articles if a.frontmatter.kind == "stub")))
    t.add_row("Bridge articles:", str(sum(1 for a in articles if a.frontmatter.kind == "bridge")))

    if map_state:
        t.add_row("Layout version:", str(map_state.layout_version))
        t.add_row("Clusters:", str(len(map_state.clusters)))
        t.add_row("Confluence zones:", str(len(map_state.confluence_zones)))
        t.add_row("Last evolved:", map_state.last_evolved.strftime("%Y-%m-%d %H:%M UTC"))
        t.add_row("Embedding model:", map_state.embedding_model)
    else:
        t.add_row("Map state:", "[yellow]none — run evolve[/yellow]")

    console.print(t)

    # Suggest evolve if thresholds crossed
    if articles:
        _maybe_suggest_evolve(config)  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _maybe_suggest_evolve(config: object) -> None:
    from wikiscapes.core.evolve import should_suggest_evolve
    should, reason = should_suggest_evolve(config)  # type: ignore
    if should:
        console.print()
        console.print(f"[yellow]ℹ[/yellow] {reason}")
        console.print("  Run [cyan]wikiscapes evolve[/cyan] to update the topographic map.")


_CLAUDE_MD_TEMPLATE = """\
# CLAUDE.md — Wikiscapes Schema Governance

This file governs how the LLM maintains this wiki. Read it before every ingest or query.

## Article Structure

Every article in `wiki/` must have YAML frontmatter with:
- `id`: stable slug matching the filename
- `title`: clear, encyclopedic title
- `sources`: list of paths to raw source files
- `kind`: one of `factual`, `synthesized`, `bridge`, `stub`

## Writing Style

- Third-person encyclopedic tone
- 300–800 words unless the source demands more
- Use `##` subheadings to organize content
- End every article with `## Key Concepts` (5–10 bullet points)
- Do not invent facts beyond the source material

## Ingest Rules

1. Check `wiki/index.md` before creating a new article
2. If a closely related article already exists, update it instead (respond UPDATE:<id>)
3. Preserve source provenance — always list source files

## Query Rules

1. Answer from provided articles only
2. Cite article IDs in brackets: [article-id]
3. Flag knowledge gaps explicitly with GAP_DETECTED:

## Do Not Edit Manually

- `wiki/index.md` — auto-maintained by wikiscapes
- `topo:` frontmatter fields — managed by the layout engine
- `connections:` frontmatter fields — managed by the spatial engine
- `access:` frontmatter fields — managed by the query engine
"""


if __name__ == "__main__":
    app()
