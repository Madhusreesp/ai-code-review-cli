"""
Pydantic v2 schemas for the AI Code Review CLI.

Defines the structured output models used to parse AI responses
and represent review results throughout the application.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class Issue(BaseModel):
    """Represents a single code issue found during review."""

    line_number: int = Field(
        ...,
        description="Line number in the file where the issue was found",
        ge=1,
    )
    severity: Literal["critical", "warning", "info"] = Field(
        ...,
        description="Severity level of the issue",
    )
    issue_type: Literal["bug", "security", "style", "performance"] = Field(
        ...,
        description="Category of the issue",
    )
    description: str = Field(
        ...,
        description="Clear explanation of what the issue is",
        min_length=10,
    )
    suggestion: str = Field(
        ...,
        description="Actionable suggestion to fix or improve the code",
        min_length=10,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "line_number": 42,
                "severity": "critical",
                "issue_type": "security",
                "description": "User input is passed directly to a SQL query without sanitization.",
                "suggestion": "Use parameterized queries or an ORM to prevent SQL injection.",
            }
        }
    }


class CodeReviewResult(BaseModel):
    """Aggregated code review result for a single file."""

    file_name: str = Field(
        ...,
        description="Path of the reviewed file relative to the repository root",
    )
    issues: List[Issue] = Field(
        default_factory=list,
        description="List of issues found in the file",
    )
    overall_quality_score: int = Field(
        ...,
        description="Overall code quality score from 1 (poor) to 10 (excellent)",
        ge=1,
        le=10,
    )
    summary: str = Field(
        ...,
        description="High-level summary of the review findings",
        min_length=10,
    )

    @field_validator("overall_quality_score")
    @classmethod
    def clamp_score(cls, v: int) -> int:
        """Ensure the score stays within the valid 1-10 range."""
        return max(1, min(10, v))

    @property
    def critical_issues(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "critical"]

    @property
    def has_critical_issues(self) -> bool:
        return len(self.critical_issues) > 0

    @property
    def issue_count_by_severity(self) -> dict[str, int]:
        counts: dict[str, int] = {"critical": 0, "warning": 0, "info": 0}
        for issue in self.issues:
            counts[issue.severity] += 1
        return counts

    @property
    def issue_count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {"bug": 0, "security": 0, "style": 0, "performance": 0}
        for issue in self.issues:
            counts[issue.issue_type] += 1
        return counts

    model_config = {
        "json_schema_extra": {
            "example": {
                "file_name": "src/auth/login.py",
                "issues": [
                    {
                        "line_number": 42,
                        "severity": "critical",
                        "issue_type": "security",
                        "description": "User input passed directly to SQL query.",
                        "suggestion": "Use parameterized queries.",
                    }
                ],
                "overall_quality_score": 6,
                "summary": "The file has one critical security issue that must be addressed.",
            }
        }
    }


class ReviewSession(BaseModel):
    """Holds the results of reviewing multiple files in one session."""

    pr_number: int | None = Field(None, description="GitHub PR number if applicable")
    repo: str | None = Field(None, description="GitHub repository in owner/repo format")
    results: List[CodeReviewResult] = Field(
        default_factory=list,
        description="Per-file review results",
    )

    @property
    def total_issues(self) -> int:
        return sum(len(r.issues) for r in self.results)

    @property
    def average_quality_score(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(r.overall_quality_score for r in self.results) / len(self.results), 2)

    @property
    def has_critical_issues(self) -> bool:
        return any(r.has_critical_issues for r in self.results)
