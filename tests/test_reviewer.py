"""
Tests for src/reviewer.py — AIReviewer logic with Gemini API mocked.

No real API calls are made. google.genai.Client is patched at the
point where AIReviewer instantiates it.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.diff_parser import ChangedFile
from src.models import CodeReviewResult
from src.reviewer import AIReviewer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_changed_file(file_path: str = "src/app.py", diff_text: str = "+x = 1\n") -> ChangedFile:
    return ChangedFile(
        file_path=file_path,
        old_content="",
        new_content="x = 1\n",
        diff_text=diff_text,
        changed_line_numbers=[1],
    )


def _valid_response_json(file_path: str = "src/app.py") -> str:
    return json.dumps({
        "file_name": file_path,
        "issues": [
            {
                "line_number": 1,
                "severity": "warning",
                "issue_type": "style",
                "description": "Variable name is not descriptive enough for readability.",
                "suggestion": "Rename the variable to something more meaningful.",
            }
        ],
        "overall_quality_score": 7,
        "summary": "Minor style issue found; overall the code is acceptable.",
    })


def _make_reviewer(model_name: str = "gemini-2.5-flash") -> AIReviewer:
    """Return an AIReviewer with genai.Client patched out."""
    with patch("src.reviewer.genai.Client"):
        reviewer = AIReviewer(api_key="fake-key", model=model_name)
    return reviewer


def _make_mock_response(json_text: str) -> MagicMock:
    """Build a mock response matching the new SDK's response structure."""
    part = MagicMock()
    part.text = json_text
    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    response = MagicMock()
    response.candidates = [candidate]
    return response


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_valid_json_parses_correctly(self):
        raw = _valid_response_json()
        result = AIReviewer._parse_response(raw, "src/app.py")
        assert isinstance(result, CodeReviewResult)
        assert result.file_name == "src/app.py"
        assert result.overall_quality_score == 7
        assert len(result.issues) == 1

    def test_file_name_defaulted_from_argument(self):
        data = json.loads(_valid_response_json())
        del data["file_name"]
        result = AIReviewer._parse_response(json.dumps(data), "src/override.py")
        assert result.file_name == "src/override.py"

    def test_string_line_number_coerced_to_int(self):
        data = json.loads(_valid_response_json())
        data["issues"][0]["line_number"] = "5"
        result = AIReviewer._parse_response(json.dumps(data), "f.py")
        assert result.issues[0].line_number == 5

    def test_invalid_line_number_string_defaults_to_1(self):
        data = json.loads(_valid_response_json())
        data["issues"][0]["line_number"] = "not-a-number"
        result = AIReviewer._parse_response(json.dumps(data), "f.py")
        assert result.issues[0].line_number == 1

    def test_markdown_fences_stripped(self):
        raw = "```json\n" + _valid_response_json() + "\n```"
        result = AIReviewer._parse_response(raw, "src/app.py")
        assert result.overall_quality_score == 7

    def test_invalid_json_raises_validation_error(self):
        with pytest.raises(ValidationError):
            AIReviewer._parse_response("not valid json {{{", "f.py")

    def test_missing_required_field_raises_validation_error(self):
        data = json.loads(_valid_response_json())
        del data["overall_quality_score"]
        with pytest.raises(ValidationError):
            AIReviewer._parse_response(json.dumps(data), "f.py")


# ---------------------------------------------------------------------------
# _fallback_result
# ---------------------------------------------------------------------------

class TestFallbackResult:
    def test_returns_valid_code_review_result(self):
        result = AIReviewer._fallback_result("src/broken.py", "timeout error")
        assert isinstance(result, CodeReviewResult)
        assert result.file_name == "src/broken.py"
        assert result.overall_quality_score == 5
        assert len(result.issues) == 1
        assert "timeout error" in result.issues[0].description

    def test_error_message_truncated_to_200_chars(self):
        long_error = "x" * 500
        result = AIReviewer._fallback_result("f.py", long_error)
        assert len(result.issues[0].description) <= 300  # prefix + 200 chars


# ---------------------------------------------------------------------------
# _truncate_diff
# ---------------------------------------------------------------------------

class TestTruncateDiff:
    def test_short_diff_unchanged(self):
        diff = "+x = 1\n" * 10
        assert AIReviewer._truncate_diff(diff) == diff

    def test_long_diff_truncated(self):
        diff = "+x = 1\n" * 2000  # well over 12_000 chars
        result = AIReviewer._truncate_diff(diff)
        assert len(result) < len(diff)
        assert "truncated" in result

    def test_truncation_cuts_at_newline(self):
        diff = "+x = 1\n" * 2000
        result = AIReviewer._truncate_diff(diff)
        # Split off everything before the truncation notice
        truncated_body = result.split("\n\n[...")[0]
        # Every non-empty line in the truncated body must be a complete '+x = 1'
        # — no line should be cut mid-way through
        non_empty_lines = [l for l in truncated_body.splitlines() if l]
        assert all(l == "+x = 1" for l in non_empty_lines)

    def test_custom_max_chars_respected(self):
        diff = "+" + "a" * 200 + "\n"
        result = AIReviewer._truncate_diff(diff, max_chars=50)
        assert len(result) < 300


# ---------------------------------------------------------------------------
# review_file — integration with mocked Gemini client
# ---------------------------------------------------------------------------

class TestReviewFile:
    def test_successful_review(self):
        reviewer = _make_reviewer()
        reviewer._client.models.generate_content.return_value = (
            _make_mock_response(_valid_response_json())
        )

        cf = _make_changed_file()
        result = reviewer.review_file(cf)

        assert isinstance(result, CodeReviewResult)
        assert result.overall_quality_score == 7
        reviewer._client.models.generate_content.assert_called_once()

    def test_returns_fallback_after_all_retries_exhausted(self):
        from google.genai.errors import ClientError
        reviewer = _make_reviewer()
        # ClientError 429 is the rate-limit equivalent in the new SDK
        reviewer._client.models.generate_content.side_effect = ClientError(
            429, {"message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}
        )

        cf = _make_changed_file()
        result = reviewer.review_file(cf)

        assert isinstance(result, CodeReviewResult)
        assert result.overall_quality_score == 5
        assert reviewer._client.models.generate_content.call_count == reviewer.max_retries

    def test_server_error_triggers_retry(self):
        from google.genai.errors import ServerError
        reviewer = _make_reviewer()
        good_response = _make_mock_response(_valid_response_json())
        reviewer._client.models.generate_content.side_effect = [
            ServerError(503, {"message": "service unavailable", "status": "UNAVAILABLE"}),
            good_response,
        ]

        cf = _make_changed_file()
        result = reviewer.review_file(cf)

        assert result.overall_quality_score == 7
        assert reviewer._client.models.generate_content.call_count == 2

    def test_validation_error_triggers_retry(self):
        reviewer = _make_reviewer()
        bad_response = _make_mock_response('{"bad": "data"}')
        good_response = _make_mock_response(_valid_response_json())
        reviewer._client.models.generate_content.side_effect = [bad_response, good_response]

        cf = _make_changed_file()
        result = reviewer.review_file(cf)

        assert result.overall_quality_score == 7
        assert reviewer._client.models.generate_content.call_count == 2

    def test_non_retryable_api_error_raises(self):
        from google.genai.errors import APIError
        reviewer = _make_reviewer()
        reviewer._client.models.generate_content.side_effect = APIError(
            401, {"message": "invalid API key", "status": "UNAUTHENTICATED"}
        )

        cf = _make_changed_file()
        with pytest.raises(APIError):
            reviewer.review_file(cf)


# ---------------------------------------------------------------------------
# review_session
# ---------------------------------------------------------------------------

class TestReviewSession:
    def test_skips_non_reviewable_files(self):
        reviewer = _make_reviewer()
        non_reviewable = ChangedFile(
            file_path="README.md",
            old_content="",
            new_content="# Hello",
            diff_text="+# Hello\n",
            changed_line_numbers=[1],
        )
        session = reviewer.review_session([non_reviewable])
        assert session.results == []
        reviewer._client.models.generate_content.assert_not_called()

    def test_reviews_all_reviewable_files(self):
        reviewer = _make_reviewer()
        reviewer._client.models.generate_content.side_effect = [
            _make_mock_response(_valid_response_json("a.py")),
            _make_mock_response(_valid_response_json("b.py")),
        ]

        files = [_make_changed_file("a.py"), _make_changed_file("b.py")]
        session = reviewer.review_session(files)

        assert len(session.results) == 2
        assert reviewer._client.models.generate_content.call_count == 2
