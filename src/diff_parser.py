"""
Git diff parser using GitPython.

Extracts changed files and their modified line ranges from a local
git repository to feed into the AI review pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import git
from git import InvalidGitRepositoryError, NoSuchPathError, Repo

logger = logging.getLogger(__name__)

# File extensions considered reviewable source code
REVIEWABLE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".kt", ".go", ".rs",
        ".c", ".cpp", ".h", ".hpp",
        ".rb", ".php", ".cs", ".swift",
        ".sh", ".bash",
    }
)


@dataclass
class ChangedFile:
    """Represents a single changed file extracted from a git diff."""

    file_path: str
    old_content: str
    new_content: str
    diff_text: str
    changed_line_numbers: list[int] = field(default_factory=list)

    @property
    def extension(self) -> str:
        return Path(self.file_path).suffix.lower()

    @property
    def is_reviewable(self) -> bool:
        return self.extension in REVIEWABLE_EXTENSIONS

    @property
    def change_summary(self) -> str:
        added = sum(1 for line in self.diff_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in self.diff_text.splitlines() if line.startswith("-") and not line.startswith("---"))
        return f"+{added}/-{removed} lines in {self.file_path}"


class DiffParser:
    """Parses git diffs for a local repository."""

    def __init__(self, repo_path: str = ".") -> None:
        self.repo_path = Path(repo_path).resolve()
        self._repo: Repo | None = None

    # ------------------------------------------------------------------
    # Repo access
    # ------------------------------------------------------------------

    @property
    def repo(self) -> Repo:
        if self._repo is None:
            try:
                self._repo = Repo(self.repo_path, search_parent_directories=True)
                logger.debug("Opened git repo at %s", self._repo.working_dir)
            except (InvalidGitRepositoryError, NoSuchPathError) as exc:
                raise RuntimeError(
                    f"No git repository found at '{self.repo_path}'. "
                    "Make sure you're running from inside a git repo."
                ) from exc
        return self._repo

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_changed_files_vs_branch(
        self,
        base_branch: str = "main",
    ) -> list[ChangedFile]:
        """Return changed files between the current branch and *base_branch*."""
        repo = self.repo
        try:
            base_commit = repo.commit(base_branch)
        except git.BadName:
            # Fall back to 'master' if 'main' doesn't exist
            try:
                base_commit = repo.commit("master")
                logger.warning("Branch 'main' not found, falling back to 'master'")
            except git.BadName as exc:
                raise RuntimeError(
                    f"Base branch '{base_branch}' (and 'master') not found in this repository."
                ) from exc

        head_commit = repo.head.commit
        diff_index = base_commit.diff(head_commit, create_patch=True)
        logger.info(
            "Comparing %s..HEAD (%d diff entries)",
            base_branch,
            len(diff_index),
        )
        return list(self._parse_diff_index(diff_index))

    def get_staged_changes(self) -> list[ChangedFile]:
        """Return files that are currently staged (index vs HEAD)."""
        repo = self.repo
        diff_index = repo.index.diff(repo.head.commit, create_patch=True)
        logger.info("Found %d staged diff entries", len(diff_index))
        return list(self._parse_diff_index(diff_index))

    def get_unstaged_changes(self) -> list[ChangedFile]:
        """Return files with unstaged working-tree changes."""
        repo = self.repo
        diff_index = repo.index.diff(None, create_patch=True)
        logger.info("Found %d unstaged diff entries", len(diff_index))
        return list(self._parse_diff_index(diff_index))

    def parse_diff_text(self, diff_text: str, file_path: str = "unknown") -> ChangedFile:
        """Parse a raw unified-diff string (e.g. from a GitHub API response)."""
        changed_lines = list(self._extract_changed_line_numbers(diff_text))
        return ChangedFile(
            file_path=file_path,
            old_content="",
            new_content="",
            diff_text=diff_text,
            changed_line_numbers=changed_lines,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_diff_index(
        self,
        diff_index: git.DiffIndex,  # type: ignore[type-arg]
    ) -> Iterator[ChangedFile]:
        for diff_item in diff_index:
            file_path: str = diff_item.b_path or diff_item.a_path or "unknown"

            # Skip binary and non-reviewable files
            if diff_item.diff and b"\x00" in diff_item.diff:
                logger.debug("Skipping binary file: %s", file_path)
                continue

            diff_text = self._decode_diff(diff_item.diff)
            old_content = self._decode_blob(diff_item.a_blob)
            new_content = self._decode_blob(diff_item.b_blob)
            changed_lines = list(self._extract_changed_line_numbers(diff_text))

            changed_file = ChangedFile(
                file_path=file_path,
                old_content=old_content,
                new_content=new_content,
                diff_text=diff_text,
                changed_line_numbers=changed_lines,
            )

            if not changed_file.is_reviewable:
                logger.debug("Skipping non-reviewable file: %s", file_path)
                continue

            logger.debug("Parsed diff for %s", changed_file.change_summary)
            yield changed_file

    @staticmethod
    def _decode_diff(raw: bytes | None) -> str:
        if not raw:
            return ""
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return ""

    @staticmethod
    def _decode_blob(blob: git.Blob | None) -> str:  # type: ignore[name-defined]
        if blob is None:
            return ""
        try:
            return blob.data_stream.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    @staticmethod
    def _extract_changed_line_numbers(diff_text: str) -> Iterator[int]:
        """
        Parse unified diff hunk headers to extract the new-file line numbers
        that were added or modified.

        Hunk header format:  @@ -old_start,old_count +new_start,new_count @@
        """
        current_line = 0
        for raw_line in diff_text.splitlines():
            if raw_line.startswith("@@"):
                # e.g.  "@@ -10,7 +10,9 @@"
                try:
                    new_part = raw_line.split("+")[1].split("@@")[0].strip()
                    start = int(new_part.split(",")[0])
                    current_line = start
                except (IndexError, ValueError):
                    current_line = 0
            elif raw_line.startswith("+") and not raw_line.startswith("+++"):
                if current_line > 0:
                    yield current_line
                current_line += 1
            elif not raw_line.startswith("-"):
                current_line += 1


def parse_local_file(file_path: str) -> ChangedFile:
    """
    Helper to wrap a local file for review without a git diff.
    The entire file content is treated as the diff/new content.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = path.read_text(encoding="utf-8", errors="replace")
    line_numbers = list(range(1, len(content.splitlines()) + 1))

    # Build a synthetic unified diff so the rest of the pipeline
    # can treat local-file reviews the same as diff-based reviews.
    diff_lines = ["--- /dev/null", f"+++ {file_path}"]
    diff_lines.append(f"@@ -0,0 +1,{len(line_numbers)} @@")
    diff_lines.extend(f"+{line}" for line in content.splitlines())
    synthetic_diff = "\n".join(diff_lines)

    return ChangedFile(
        file_path=str(path),
        old_content="",
        new_content=content,
        diff_text=synthetic_diff,
        changed_line_numbers=line_numbers,
    )
