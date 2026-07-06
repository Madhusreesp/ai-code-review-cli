"""Tests for src/cli.py — evaluate_and_exit, --file mode, argument parsing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from src.cli import app, evaluate_and_exit
from src.models import CodeReviewResult, Issue, ReviewSession

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_issue(**overrides) -> Issue:
    base = dict(
        line_number=5,
        severity="warning",
        issue_type="bug",
        description="Variable is never used after assignment.",
        suggestion="Remove the unused variable or use it appropriately.",
    )
    base.update(overrides)
    return Issue(**base)


def _make_result(issues=None, score=8) -> CodeReviewResult:
    return CodeReviewResult(
        file_name="src/app.py",
        issues=issues or [],
        overall_quality_score=score,
        summary="The file looks clean with no major issues found.",
    )


def _make_session(has_critical: bool = False) -> ReviewSession:
    if has_critical:
        issues = [_make_issue(severity="critical")]
    else:
        issues = [_make_issue(severity="warning")]
    return ReviewSession(results=[_make_result(issues=issues)])


# ---------------------------------------------------------------------------
# evaluate_and_exit
# ---------------------------------------------------------------------------

class TestEvaluateAndExit:
    def test_exits_1_when_critical_issues(self):
        session = _make_session(has_critical=True)
        with pytest.raises(typer.Exit) as exc_info:
            evaluate_and_exit(session)
        assert exc_info.value.exit_code == 1

    def test_exits_0_when_no_critical_issues(self):
        session = _make_session(has_critical=False)
        with pytest.raises(typer.Exit) as exc_info:
            evaluate_and_exit(session)
        assert exc_info.value.exit_code == 0

    def test_exits_0_for_empty_session(self):
        session = ReviewSession(results=[])
        with pytest.raises(typer.Exit) as exc_info:
            evaluate_and_exit(session)
        assert exc_info.value.exit_code == 0


# ---------------------------------------------------------------------------
# --file mode: never calls evaluate_and_exit
# ---------------------------------------------------------------------------

class TestFileModeNoExit:
    def _invoke_file_mode(self, has_critical: bool, tmp_path) -> int:
        mock_session = _make_session(has_critical=has_critical)
        mock_changed_file = MagicMock()
        # Create a real file so Path(...).exists() returns True
        real_file = tmp_path / "app.py"
        real_file.write_text("x = 1\n")

        with (
            patch("src.cli._get_openai_key", return_value="fake-key"),
            patch("src.cli.AIReviewer") as MockReviewer,
            patch("src.cli.parse_local_file", return_value=mock_changed_file),
        ):
            mock_reviewer = MockReviewer.return_value
            mock_reviewer.review_session.return_value = mock_session
            result = runner.invoke(app, ["--file", str(real_file)])
        return result.exit_code

    def test_file_mode_exits_0_with_no_critical_issues(self, tmp_path):
        assert self._invoke_file_mode(has_critical=False, tmp_path=tmp_path) == 0

    def test_file_mode_exits_0_even_with_critical_issues(self, tmp_path):
        # --file mode must NOT call evaluate_and_exit, so exit code is always 0
        assert self._invoke_file_mode(has_critical=True, tmp_path=tmp_path) == 0


# ---------------------------------------------------------------------------
# --diff mode: calls evaluate_and_exit
# ---------------------------------------------------------------------------

class TestDiffModeExit:
    def _invoke_diff_mode(self, has_critical: bool) -> int:
        mock_session = _make_session(has_critical=has_critical)
        mock_changed_file = MagicMock()

        with (
            patch("src.cli._get_openai_key", return_value="fake-key"),
            patch("src.cli.AIReviewer") as MockReviewer,
            patch("src.cli.DiffParser") as MockParser,
        ):
            mock_parser = MockParser.return_value
            mock_parser.get_changed_files_vs_branch.return_value = [mock_changed_file]

            mock_reviewer = MockReviewer.return_value
            mock_reviewer.review_session.return_value = mock_session

            result = runner.invoke(app, ["--diff"])
        return result.exit_code

    def test_diff_mode_exits_1_with_critical_issues(self):
        assert self._invoke_diff_mode(has_critical=True) == 1

    def test_diff_mode_exits_0_without_critical_issues(self):
        assert self._invoke_diff_mode(has_critical=False) == 0


# ---------------------------------------------------------------------------
# Argument parsing / missing required args
# ---------------------------------------------------------------------------

class TestArgumentParsing:
    def test_no_args_exits_nonzero(self):
        with patch("src.cli._get_openai_key", return_value="fake-key"):
            result = runner.invoke(app, [])
        assert result.exit_code != 0

    def test_pr_number_without_repo_exits_nonzero(self):
        with (
            patch("src.cli._get_openai_key", return_value="fake-key"),
            patch("src.cli._get_github_token", return_value="fake-token"),
            patch("src.cli.GitHubClient"),
        ):
            result = runner.invoke(app, ["--pr-number", "42"])
        assert result.exit_code != 0

    def test_pr_number_without_repo_shows_error_message(self):
        with (
            patch("src.cli._get_openai_key", return_value="fake-key"),
            patch("src.cli._get_github_token", return_value="fake-token"),
            patch("src.cli.GitHubClient"),
        ):
            result = runner.invoke(app, ["--pr-number", "42"])
        assert "repo" in result.output.lower() or result.exit_code != 0

    def test_help_flag_exits_zero(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_help_output_mentions_pr_number(self):
        result = runner.invoke(app, ["--help"])
        assert "--pr-number" in result.output
