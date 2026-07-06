"""Tests for src/models.py — Issue, CodeReviewResult, ReviewSession."""

import pytest
from pydantic import ValidationError

from src.models import CodeReviewResult, Issue, ReviewSession


# ---------------------------------------------------------------------------
# Issue
# ---------------------------------------------------------------------------

def _valid_issue(**overrides) -> dict:
    base = {
        "line_number": 10,
        "severity": "warning",
        "issue_type": "bug",
        "description": "Variable is never used after assignment.",
        "suggestion": "Remove the unused variable or use it appropriately.",
    }
    base.update(overrides)
    return base


class TestIssue:
    def test_valid_construction(self):
        issue = Issue(**_valid_issue())
        assert issue.line_number == 10
        assert issue.severity == "warning"
        assert issue.issue_type == "bug"

    def test_all_severities_accepted(self):
        for sev in ("critical", "warning", "info"):
            issue = Issue(**_valid_issue(severity=sev))
            assert issue.severity == sev

    def test_all_issue_types_accepted(self):
        for itype in ("bug", "security", "style", "performance"):
            issue = Issue(**_valid_issue(issue_type=itype))
            assert issue.issue_type == itype

    def test_line_number_must_be_positive(self):
        with pytest.raises(ValidationError):
            Issue(**_valid_issue(line_number=0))

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValidationError):
            Issue(**_valid_issue(severity="blocker"))

    def test_invalid_issue_type_rejected(self):
        with pytest.raises(ValidationError):
            Issue(**_valid_issue(issue_type="typo"))

    def test_description_min_length_enforced(self):
        with pytest.raises(ValidationError):
            Issue(**_valid_issue(description="Too short"))

    def test_suggestion_min_length_enforced(self):
        with pytest.raises(ValidationError):
            Issue(**_valid_issue(suggestion="Fix it."))


# ---------------------------------------------------------------------------
# CodeReviewResult
# ---------------------------------------------------------------------------

def _make_result(**overrides) -> CodeReviewResult:
    base = dict(
        file_name="src/app.py",
        issues=[],
        overall_quality_score=8,
        summary="The file looks clean with no major issues found.",
    )
    base.update(overrides)
    return CodeReviewResult(**base)


class TestCodeReviewResult:
    def test_valid_construction(self):
        result = _make_result()
        assert result.file_name == "src/app.py"
        assert result.overall_quality_score == 8
        assert result.issues == []

    def test_score_above_10_rejected(self):
        # ge/le field constraints reject out-of-range values with ValidationError
        with pytest.raises(ValidationError):
            _make_result(overall_quality_score=15)

    def test_score_below_1_rejected(self):
        with pytest.raises(ValidationError):
            _make_result(overall_quality_score=-3)

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            CodeReviewResult(
                file_name="f.py",
                overall_quality_score=0,
                summary="A sufficiently long summary string here.",
            )

    def test_has_critical_issues_false_when_empty(self):
        result = _make_result()
        assert result.has_critical_issues is False

    def test_has_critical_issues_true(self):
        critical = Issue(**_valid_issue(severity="critical"))
        result = _make_result(issues=[critical])
        assert result.has_critical_issues is True

    def test_critical_issues_property_filters_correctly(self):
        issues = [
            Issue(**_valid_issue(severity="critical")),
            Issue(**_valid_issue(severity="warning")),
            Issue(**_valid_issue(severity="info")),
        ]
        result = _make_result(issues=issues)
        assert len(result.critical_issues) == 1
        assert result.critical_issues[0].severity == "critical"

    def test_issue_count_by_severity(self):
        issues = [
            Issue(**_valid_issue(severity="critical")),
            Issue(**_valid_issue(severity="critical")),
            Issue(**_valid_issue(severity="warning")),
        ]
        result = _make_result(issues=issues)
        counts = result.issue_count_by_severity
        assert counts["critical"] == 2
        assert counts["warning"] == 1
        assert counts["info"] == 0

    def test_issue_count_by_type(self):
        issues = [
            Issue(**_valid_issue(issue_type="bug")),
            Issue(**_valid_issue(issue_type="security")),
            Issue(**_valid_issue(issue_type="security")),
        ]
        result = _make_result(issues=issues)
        counts = result.issue_count_by_type
        assert counts["bug"] == 1
        assert counts["security"] == 2
        assert counts["style"] == 0

    def test_summary_min_length_enforced(self):
        with pytest.raises(ValidationError):
            CodeReviewResult(
                file_name="f.py",
                overall_quality_score=5,
                summary="Short",
            )


# ---------------------------------------------------------------------------
# ReviewSession
# ---------------------------------------------------------------------------

class TestReviewSession:
    def test_empty_session(self):
        session = ReviewSession()
        assert session.total_issues == 0
        assert session.average_quality_score == 0.0
        assert session.has_critical_issues is False

    def test_total_issues_sums_across_files(self):
        r1 = _make_result(issues=[Issue(**_valid_issue()), Issue(**_valid_issue())])
        r2 = _make_result(issues=[Issue(**_valid_issue())])
        session = ReviewSession(results=[r1, r2])
        assert session.total_issues == 3

    def test_average_quality_score(self):
        r1 = _make_result(overall_quality_score=6)
        r2 = _make_result(overall_quality_score=8)
        session = ReviewSession(results=[r1, r2])
        assert session.average_quality_score == 7.0

    def test_has_critical_issues_propagates(self):
        clean = _make_result()
        critical = _make_result(issues=[Issue(**_valid_issue(severity="critical"))])
        session = ReviewSession(results=[clean, critical])
        assert session.has_critical_issues is True

    def test_pr_number_and_repo_optional(self):
        session = ReviewSession(pr_number=42, repo="owner/repo")
        assert session.pr_number == 42
        assert session.repo == "owner/repo"
