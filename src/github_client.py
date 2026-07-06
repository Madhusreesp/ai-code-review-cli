"""
GitHub API integration using PyGithub.

Handles fetching PR diffs and posting inline review comments
and summary comments back to a pull request.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import requests
from github import Github, GithubException
from github.PullRequest import PullRequest
from github.Repository import Repository

if TYPE_CHECKING:
    from src.models import CodeReviewResult, ReviewSession

logger = logging.getLogger(__name__)

# Severity → GitHub review comment emoji mapping
SEVERITY_EMOJI: dict[str, str] = {
    "critical": "🔴",
    "warning":  "🟡",
    "info":     "🔵",
}

ISSUE_TYPE_EMOJI: dict[str, str] = {
    "bug":         "🐛",
    "security":    "🔒",
    "style":       "✨",
    "performance": "⚡",
}

SCORE_EMOJI: dict[range, str] = {
    range(1, 4):  "🔴",  # 1-3  poor
    range(4, 7):  "🟡",  # 4-6  average
    range(7, 11): "🟢",  # 7-10 good
}


def _score_emoji(score: int) -> str:
    for r, emoji in SCORE_EMOJI.items():
        if score in r:
            return emoji
    return "⚪"


class GitHubClient:
    """Wrapper around PyGithub for PR review workflows."""

    def __init__(self, token: str, repo_name: str) -> None:
        """
        Args:
            token:     GitHub personal access token or GITHUB_TOKEN.
            repo_name: Repository in 'owner/repo' format.
        """
        self._gh = Github(token)
        self._repo_name = repo_name
        self._repo: Repository | None = None
        self._token = token

    # ------------------------------------------------------------------
    # Repo / PR helpers
    # ------------------------------------------------------------------

    @property
    def repo(self) -> Repository:
        if self._repo is None:
            try:
                self._repo = self._gh.get_repo(self._repo_name)
                logger.debug("Connected to GitHub repo: %s", self._repo_name)
            except GithubException as exc:
                raise RuntimeError(
                    f"Cannot access repository '{self._repo_name}': {exc.data.get('message', exc)}"
                ) from exc
        return self._repo

    def get_pull_request(self, pr_number: int) -> PullRequest:
        try:
            pr = self.repo.get_pull(pr_number)
            logger.info("Fetched PR #%d: %s", pr_number, pr.title)
            return pr
        except GithubException as exc:
            raise RuntimeError(
                f"Cannot fetch PR #{pr_number} from '{self._repo_name}': "
                f"{exc.data.get('message', exc)}"
            ) from exc

    def get_pr_diff_files(self, pr_number: int) -> list[dict[str, str]]:
        """
        Return a list of dicts with 'filename' and 'patch' keys
        for each changed file in the PR.
        """
        pr = self.get_pull_request(pr_number)
        files = []
        for f in pr.get_files():
            if f.patch:  # binary files have no patch
                files.append(
                    {
                        "filename": f.filename,
                        "patch": f.patch,
                        "status": f.status,
                        "additions": f.additions,
                        "deletions": f.deletions,
                    }
                )
        logger.info(
            "PR #%d has %d reviewable file(s)",
            pr_number,
            len(files),
        )
        return files

    # ------------------------------------------------------------------
    # Posting review comments
    # ------------------------------------------------------------------

    def post_review(self, pr_number: int, session: "ReviewSession") -> None:
        """
        Post a full code review to a PR:
        - One inline comment per Issue (at the specific line).
        - One summary comment with the overall session results.
        """
        pr = self.get_pull_request(pr_number)
        commit = pr.get_commits().reversed[0]  # latest commit

        inline_comments: list[dict] = []
        for result in session.results:
            for issue in result.issues:
                body = self._format_inline_comment(issue)
                inline_comments.append(
                    {
                        "path": result.file_name,
                        "line": issue.line_number,
                        "body": body,
                    }
                )

        review_body = self._format_summary_comment(session)
        review_event = "REQUEST_CHANGES" if session.has_critical_issues else "COMMENT"

        try:
            pr.create_review(
                commit=commit,
                body=review_body,
                event=review_event,
                comments=inline_comments,
            )
            logger.info(
                "Posted review on PR #%d with %d inline comment(s)",
                pr_number,
                len(inline_comments),
            )
        except GithubException as exc:
            logger.error("Failed to post review: %s", exc)
            # Fallback: post a plain issue comment so the summary is not lost
            self._post_fallback_comment(pr, review_body)

    def post_summary_comment(self, pr_number: int, session: "ReviewSession") -> None:
        """Post only the summary comment (no inline comments)."""
        pr = self.get_pull_request(pr_number)
        body = self._format_summary_comment(session)
        try:
            pr.create_issue_comment(body)
            logger.info("Posted summary comment on PR #%d", pr_number)
        except GithubException as exc:
            raise RuntimeError(f"Failed to post summary comment: {exc}") from exc

    def post_inline_comments(self, pr_number: int, session: "ReviewSession") -> None:
        """
        Post each Issue as an individual inline PR review comment at its
        exact line number, then post the summary as a separate issue comment.

        Uses pr.create_review() so all inline comments are submitted in a
        single API call, which avoids GitHub's per-file comment rate limits.
        Falls back to post_summary_comment() if the review API call fails.
        """
        pr = self.get_pull_request(pr_number)
        commit = pr.get_commits().reversed[0]  # latest commit on the PR

        inline_comments: list[dict] = []
        for result in session.results:
            for issue in result.issues:
                inline_comments.append(
                    {
                        "path": result.file_name,
                        "line": issue.line_number,
                        "body": self._format_inline_comment(issue),
                    }
                )

        review_event = "REQUEST_CHANGES" if session.has_critical_issues else "COMMENT"
        summary_body = self._format_summary_comment(session)

        try:
            pr.create_review(
                commit=commit,
                body=summary_body,
                event=review_event,
                comments=inline_comments,
            )
            logger.info(
                "Posted inline review on PR #%d: %d comment(s), event=%s",
                pr_number,
                len(inline_comments),
                review_event,
            )
        except GithubException as exc:
            logger.error(
                "create_review() failed for PR #%d (%s); falling back to summary comment",
                pr_number,
                exc,
            )
            self._post_fallback_comment(pr, summary_body)

    # ------------------------------------------------------------------
    # Comment formatters
    # ------------------------------------------------------------------

    @staticmethod
    def _format_inline_comment(issue) -> str:  # type: ignore[no-untyped-def]
        sev_emoji = SEVERITY_EMOJI.get(issue.severity, "⚪")
        type_emoji = ISSUE_TYPE_EMOJI.get(issue.issue_type, "🔍")
        return (
            f"{sev_emoji} **{issue.severity.upper()}** {type_emoji} `{issue.issue_type}`\n\n"
            f"**Issue:** {issue.description}\n\n"
            f"**Suggestion:** {issue.suggestion}"
        )

    @staticmethod
    def _format_summary_comment(session: "ReviewSession") -> str:
        lines: list[str] = [
            "## 🤖 AI Code Review Summary",
            "",
            f"**Overall average quality score:** "
            f"{_score_emoji(int(session.average_quality_score))} "
            f"{session.average_quality_score}/10",
            f"**Total issues found:** {session.total_issues}",
            "",
            "---",
            "",
        ]

        for result in session.results:
            score_emoji = _score_emoji(result.overall_quality_score)
            lines += [
                f"### 📄 `{result.file_name}`",
                f"**Quality score:** {score_emoji} {result.overall_quality_score}/10",
                f"**Summary:** {result.summary}",
                "",
            ]
            if result.issues:
                severity_counts = result.issue_count_by_severity
                lines.append(
                    f"**Issues:** "
                    f"🔴 {severity_counts['critical']} critical · "
                    f"🟡 {severity_counts['warning']} warnings · "
                    f"🔵 {severity_counts['info']} info"
                )
                lines.append("")
                lines.append("| Line | Severity | Type | Description |")
                lines.append("|------|----------|------|-------------|")
                for issue in result.issues:
                    sev_emoji = SEVERITY_EMOJI.get(issue.severity, "⚪")
                    type_emoji = ISSUE_TYPE_EMOJI.get(issue.issue_type, "🔍")
                    short_desc = (
                        issue.description[:80] + "…"
                        if len(issue.description) > 80
                        else issue.description
                    )
                    lines.append(
                        f"| {issue.line_number} | {sev_emoji} {issue.severity} | "
                        f"{type_emoji} {issue.issue_type} | {short_desc} |"
                    )
                lines.append("")
            else:
                lines.append("✅ No issues found.")
                lines.append("")

        lines += [
            "---",
            "*Generated by [AI Code Review CLI](https://github.com/your-org/ai-code-review-cli)*",
        ]
        return "\n".join(lines)

    @staticmethod
    def _post_fallback_comment(pr: PullRequest, body: str) -> None:
        """Post body as a plain issue comment when review API fails."""
        try:
            pr.create_issue_comment(body)
            logger.info("Posted fallback issue comment on PR #%d", pr.number)
        except GithubException as exc:
            logger.error("Fallback comment also failed: %s", exc)

    # ------------------------------------------------------------------
    # PR diff fetching via raw REST (alternative to PyGithub)
    # ------------------------------------------------------------------

    def fetch_pr_diff_raw(self, pr_number: int) -> str:
        """
        Fetch the raw unified diff for a PR using the GitHub REST API.
        Useful when you want the full diff as a single string.
        """
        url = f"https://api.github.com/repos/{self._repo_name}/pulls/{pr_number}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github.diff",
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.text
