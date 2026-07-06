"""Tests for src/github_client.py — GitHubClient."""

from unittest.mock import MagicMock, patch

import pytest
from github import GithubException

from src.github_client import GitHubClient
from src.models import CodeReviewResult, Issue, ReviewSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_issue(**overrides) -> Issue:
    base = dict(
        line_number=10,
        severity="warning",
        issue_type="bug",
        description="Variable is never used after assignment.",
        suggestion="Remove the unused variable or use it appropriately.",
    )
    base.update(overrides)
    return Issue(**base)


def _make_result(file_name="src/app.py", issues=None, score=8) -> CodeReviewResult:
    return CodeReviewResult(
        file_name=file_name,
        issues=issues or [],
        overall_quality_score=score,
        summary="The file looks clean with no major issues found.",
    )


def _make_session(results=None, pr_number=None, repo=None) -> ReviewSession:
    return ReviewSession(
        results=results or [],
        pr_number=pr_number,
        repo=repo,
    )


def _make_client() -> tuple[GitHubClient, MagicMock]:
    """Return (client, mock_repo) with Github patched out."""
    with patch("src.github_client.Github") as MockGithub:
        mock_repo = MagicMock()
        MockGithub.return_value.get_repo.return_value = mock_repo
        client = GitHubClient(token="fake-token", repo_name="owner/repo")
        # Force _repo so property doesn't re-call get_repo
        client._repo = mock_repo
        return client, mock_repo


# ---------------------------------------------------------------------------
# post_inline_comments — happy path
# ---------------------------------------------------------------------------

class TestPostInlineCommentsSuccess:
    def test_calls_create_review_once(self):
        client, mock_repo = _make_client()
        mock_pr = MagicMock()
        mock_commit = MagicMock()
        mock_pr.get_commits.return_value.reversed.__getitem__.return_value = mock_commit
        mock_repo.get_pull.return_value = mock_pr

        session = _make_session(
            results=[_make_result(issues=[_make_issue(line_number=5)])],
            pr_number=1,
        )
        client.post_inline_comments(pr_number=1, session=session)

        mock_pr.create_review.assert_called_once()

    def test_inline_comment_uses_correct_line_number(self):
        client, mock_repo = _make_client()
        mock_pr = MagicMock()
        mock_pr.get_commits.return_value.reversed.__getitem__.return_value = MagicMock()
        mock_repo.get_pull.return_value = mock_pr

        issue = _make_issue(line_number=42)
        session = _make_session(results=[_make_result(issues=[issue])])
        client.post_inline_comments(pr_number=1, session=session)

        _, kwargs = mock_pr.create_review.call_args
        comments = kwargs["comments"]
        assert len(comments) == 1
        assert comments[0]["line"] == 42
        assert comments[0]["path"] == "src/app.py"

    def test_multiple_issues_produce_multiple_inline_comments(self):
        client, mock_repo = _make_client()
        mock_pr = MagicMock()
        mock_pr.get_commits.return_value.reversed.__getitem__.return_value = MagicMock()
        mock_repo.get_pull.return_value = mock_pr

        issues = [_make_issue(line_number=i) for i in (1, 2, 3)]
        session = _make_session(results=[_make_result(issues=issues)])
        client.post_inline_comments(pr_number=1, session=session)

        _, kwargs = mock_pr.create_review.call_args
        assert len(kwargs["comments"]) == 3

    def test_critical_issues_set_request_changes_event(self):
        client, mock_repo = _make_client()
        mock_pr = MagicMock()
        mock_pr.get_commits.return_value.reversed.__getitem__.return_value = MagicMock()
        mock_repo.get_pull.return_value = mock_pr

        critical = _make_issue(severity="critical")
        session = _make_session(results=[_make_result(issues=[critical])])
        client.post_inline_comments(pr_number=1, session=session)

        _, kwargs = mock_pr.create_review.call_args
        assert kwargs["event"] == "REQUEST_CHANGES"

    def test_no_critical_issues_set_comment_event(self):
        client, mock_repo = _make_client()
        mock_pr = MagicMock()
        mock_pr.get_commits.return_value.reversed.__getitem__.return_value = MagicMock()
        mock_repo.get_pull.return_value = mock_pr

        session = _make_session(results=[_make_result(issues=[_make_issue(severity="warning")])])
        client.post_inline_comments(pr_number=1, session=session)

        _, kwargs = mock_pr.create_review.call_args
        assert kwargs["event"] == "COMMENT"


# ---------------------------------------------------------------------------
# post_inline_comments — fallback on GithubException
# ---------------------------------------------------------------------------

class TestPostInlineCommentsFallback:
    def test_falls_back_to_issue_comment_on_create_review_failure(self):
        client, mock_repo = _make_client()
        mock_pr = MagicMock()
        mock_pr.get_commits.return_value.reversed.__getitem__.return_value = MagicMock()
        mock_pr.create_review.side_effect = GithubException(422, {"message": "Unprocessable"}, {})
        mock_repo.get_pull.return_value = mock_pr

        session = _make_session(results=[_make_result(issues=[_make_issue()])])
        client.post_inline_comments(pr_number=1, session=session)

        mock_pr.create_issue_comment.assert_called_once()

    def test_fallback_comment_body_is_non_empty(self):
        client, mock_repo = _make_client()
        mock_pr = MagicMock()
        mock_pr.get_commits.return_value.reversed.__getitem__.return_value = MagicMock()
        mock_pr.create_review.side_effect = GithubException(500, {"message": "Server Error"}, {})
        mock_repo.get_pull.return_value = mock_pr

        session = _make_session(results=[_make_result()])
        client.post_inline_comments(pr_number=1, session=session)

        body = mock_pr.create_issue_comment.call_args[0][0]
        assert len(body) > 0


# ---------------------------------------------------------------------------
# Summary comment includes overall_quality_score
# ---------------------------------------------------------------------------

class TestSummaryComment:
    def test_summary_includes_quality_score(self):
        client, mock_repo = _make_client()
        mock_pr = MagicMock()
        mock_pr.get_commits.return_value.reversed.__getitem__.return_value = MagicMock()
        mock_repo.get_pull.return_value = mock_pr

        session = _make_session(results=[_make_result(score=7)])
        client.post_inline_comments(pr_number=1, session=session)

        _, kwargs = mock_pr.create_review.call_args
        body = kwargs["body"]
        assert "7" in body

    def test_summary_includes_per_file_score(self):
        """Each file's overall_quality_score appears in the review body."""
        client, mock_repo = _make_client()
        mock_pr = MagicMock()
        mock_pr.get_commits.return_value.reversed.__getitem__.return_value = MagicMock()
        mock_repo.get_pull.return_value = mock_pr

        session = _make_session(results=[_make_result(score=9)])
        client.post_inline_comments(pr_number=1, session=session)

        _, kwargs = mock_pr.create_review.call_args
        assert "9/10" in kwargs["body"]
