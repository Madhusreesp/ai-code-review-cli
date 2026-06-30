"""
AI Code Review CLI — entry point.

Commands
--------
  review --pr-number <n> --repo <owner/repo>   Review a GitHub pull request.
  review --file <path>                          Review a single local file.
  review --diff                                 Review local git diff vs main.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env into os.environ before any code reads environment variables.
# override=False means a real env var already set in the shell takes precedence.
load_dotenv(override=False)

import typer
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from src.diff_parser import DiffParser, parse_local_file
from src.github_client import GitHubClient
from src.models import CodeReviewResult, ReviewSession
from src.reviewer import AIReviewer

# ---------------------------------------------------------------------------
# App / Console setup
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="ai-review",
    help="🤖 AI-powered code review using OpenAI and GitHub.",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()
err_console = Console(stderr=True, style="bold red")

SEVERITY_STYLE: dict[str, str] = {
    "critical": "bold red",
    "warning":  "bold yellow",
    "info":     "bold cyan",
}

ISSUE_TYPE_STYLE: dict[str, str] = {
    "bug":         "red",
    "security":    "magenta",
    "style":       "green",
    "performance": "blue",
}

SCORE_STYLE: dict[range, str] = {
    range(1, 4):  "bold red",
    range(4, 7):  "bold yellow",
    range(7, 11): "bold green",
}


def _score_style(score: int) -> str:
    for r, style in SCORE_STYLE.items():
        if score in r:
            return style
    return "white"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            RichHandler(
                console=console,
                show_time=False,
                rich_tracebacks=True,
                markup=True,
            )
        ],
    )


def _get_openai_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        err_console.print(
            "[bold red]Error:[/bold red] GEMINI_API_KEY environment variable is not set."
        )
        raise typer.Exit(1)
    return key


def _get_github_token() -> str:
    token = os.getenv("GITHUB_TOKEN", "")
    if not token:
        err_console.print(
            "[bold red]Error:[/bold red] GITHUB_TOKEN environment variable is not set."
        )
        raise typer.Exit(1)
    return token


def _print_banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]🤖 AI Code Review CLI[/bold cyan]\n"
            "[dim]Powered by Gemini · Built with Typer & Rich[/dim]",
            border_style="cyan",
        )
    )
    console.print()


def _render_result(result: CodeReviewResult) -> None:
    """Render a single CodeReviewResult to the terminal with Rich tables."""
    score_style = _score_style(result.overall_quality_score)

    # File header panel
    console.print(
        Panel(
            f"[bold white]{result.file_name}[/bold white]\n"
            f"Quality Score: [{score_style}]{result.overall_quality_score}/10[/{score_style}]\n"
            f"[dim]{result.summary}[/dim]",
            title="📄 File Review",
            border_style="blue",
        )
    )

    if not result.issues:
        console.print("  [bold green]✅ No issues found.[/bold green]\n")
        return

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white on dark_blue",
        expand=True,
    )
    table.add_column("Line", style="dim", width=6, justify="right")
    table.add_column("Severity", width=10)
    table.add_column("Type", width=12)
    table.add_column("Description")
    table.add_column("Suggestion")

    for issue in result.issues:
        sev_style = SEVERITY_STYLE.get(issue.severity, "white")
        type_style = ISSUE_TYPE_STYLE.get(issue.issue_type, "white")
        table.add_row(
            str(issue.line_number),
            Text(issue.severity.upper(), style=sev_style),
            Text(issue.issue_type, style=type_style),
            issue.description,
            issue.suggestion,
        )

    console.print(table)
    console.print()


def _render_session_summary(session: ReviewSession) -> None:
    """Print an aggregate summary table for all reviewed files."""
    avg = session.average_quality_score
    avg_style = _score_style(int(avg))

    console.print(
        Panel(
            f"[bold white]Files reviewed:[/bold white] {len(session.results)}\n"
            f"[bold white]Total issues:[/bold white] {session.total_issues}\n"
            f"[bold white]Average quality score:[/bold white] [{avg_style}]{avg}/10[/{avg_style}]",
            title="📊 Session Summary",
            border_style="cyan",
        )
    )

    if session.has_critical_issues:
        console.print(
            "[bold red]⚠️  Critical issues detected — please address them before merging.[/bold red]\n"
        )
    else:
        console.print("[bold green]✅ No critical issues detected.[/bold green]\n")


def _run_review_and_display(
    changed_files: list,
    reviewer: AIReviewer,
    pr_number: int | None = None,
    repo: str | None = None,
    github_token: str | None = None,
) -> ReviewSession:
    """Core logic: review files, print results, optionally post to GitHub."""
    if not changed_files:
        console.print("[yellow]No reviewable files found.[/yellow]")
        raise typer.Exit(0)

    session: ReviewSession | None = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Reviewing {len(changed_files)} file(s) with AI…", total=None
        )
        session = reviewer.review_session(changed_files)
        session.pr_number = pr_number
        session.repo = repo
        progress.update(task, completed=True)

    console.print()
    for result in session.results:
        _render_result(result)

    _render_session_summary(session)

    # Post results back to GitHub if PR info is available
    if pr_number and repo and github_token:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("[cyan]Posting review to GitHub PR…", total=None)
            gh_client = GitHubClient(token=github_token, repo_name=repo)
            try:
                gh_client.post_review(pr_number=pr_number, session=session)
                console.print(
                    f"[bold green]✅ Review posted to {repo}#PR{pr_number}[/bold green]"
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]⚠️  Could not post to GitHub: {exc}[/yellow]")

    return session


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@app.command()
def review(
    pr_number: Optional[int] = typer.Option(
        None,
        "--pr-number",
        "-p",
        help="GitHub Pull Request number to review.",
        show_default=False,
    ),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="GitHub repository in [cyan]owner/repo[/cyan] format.",
        show_default=False,
    ),
    file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Local file path to review.",
        show_default=False,
        exists=False,  # we do our own existence check for nicer errors
    ),
    diff: bool = typer.Option(
        False,
        "--diff",
        "-d",
        help="Review local git diff (current branch vs main).",
    ),
    base_branch: str = typer.Option(
        "main",
        "--base",
        "-b",
        help="Base branch to diff against (default: main).",
    ),
    model: str = typer.Option(
        "gemini-2.5-flash",
        "--model",
        "-m",
        help="Gemini model to use for review.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose debug logging.",
    ),
    no_post: bool = typer.Option(
        False,
        "--no-post",
        help="Skip posting results to GitHub (dry-run mode).",
    ),
) -> None:
    """
    [bold cyan]Run an AI-powered code review.[/bold cyan]

    Modes:
    - [green]--pr-number + --repo[/green]: fetch diff from GitHub PR and optionally post results.
    - [green]--file[/green]: review a single local file.
    - [green]--diff[/green]: review your local git diff vs the base branch.
    """
    _setup_logging(verbose)
    _print_banner()

    openai_key = _get_openai_key()
    reviewer = AIReviewer(api_key=openai_key, model=model)

    # ------------------------------------------------------------------ #
    # Mode 1: GitHub PR review
    # ------------------------------------------------------------------ #
    if pr_number is not None:
        if not repo:
            err_console.print(
                "[bold red]Error:[/bold red] --repo is required when using --pr-number."
            )
            raise typer.Exit(1)

        github_token = _get_github_token()
        gh_client = GitHubClient(token=github_token, repo_name=repo)

        console.print(
            f"[dim]Fetching diff for [bold]{repo}[/bold] PR [bold]#{pr_number}[/bold]…[/dim]"
        )

        try:
            diff_files = gh_client.get_pr_diff_files(pr_number)
        except RuntimeError as exc:
            err_console.print(f"[bold red]GitHub error:[/bold red] {exc}")
            raise typer.Exit(1) from exc

        parser = DiffParser()
        changed_files = [
            parser.parse_diff_text(f["patch"], f["filename"])
            for f in diff_files
        ]

        _run_review_and_display(
            changed_files=changed_files,
            reviewer=reviewer,
            pr_number=pr_number,
            repo=repo,
            github_token=None if no_post else github_token,
        )

    # ------------------------------------------------------------------ #
    # Mode 2: Single local file
    # ------------------------------------------------------------------ #
    elif file is not None:
        file_path = Path(file)
        if not file_path.exists():
            err_console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
            raise typer.Exit(1)

        console.print(f"[dim]Reviewing local file: [bold]{file_path}[/bold]…[/dim]\n")
        try:
            changed_file = parse_local_file(str(file_path))
        except FileNotFoundError as exc:
            err_console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(1) from exc

        _run_review_and_display(
            changed_files=[changed_file],
            reviewer=reviewer,
        )

    # ------------------------------------------------------------------ #
    # Mode 3: Local git diff
    # ------------------------------------------------------------------ #
    elif diff:
        console.print(
            f"[dim]Analysing local diff against [bold]{base_branch}[/bold]…[/dim]\n"
        )
        try:
            parser = DiffParser()
            changed_files = parser.get_changed_files_vs_branch(base_branch)
        except RuntimeError as exc:
            err_console.print(f"[bold red]Git error:[/bold red] {exc}")
            raise typer.Exit(1) from exc

        _run_review_and_display(
            changed_files=changed_files,
            reviewer=reviewer,
        )

    # ------------------------------------------------------------------ #
    # No mode selected
    # ------------------------------------------------------------------ #
    else:
        console.print(
            Panel(
                "Please specify a review mode:\n\n"
                "  [green]--pr-number <n> --repo owner/repo[/green]  Review a GitHub PR\n"
                "  [green]--file <path>[/green]                       Review a local file\n"
                "  [green]--diff[/green]                              Review local git diff\n\n"
                "Run [bold]ai-review review --help[/bold] for full options.",
                title="[yellow]⚠ No mode selected[/yellow]",
                border_style="yellow",
            )
        )
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Entry-point guard
# ---------------------------------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()
