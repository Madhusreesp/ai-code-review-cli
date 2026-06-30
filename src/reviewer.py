"""
AI-powered code reviewer using the Google Gemini API.

Sends code diffs to Gemini and parses the response directly into
CodeReviewResult Pydantic models with retry logic and graceful
validation error handling.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

import google.generativeai as genai
from google.api_core.exceptions import (
    DeadlineExceeded,
    ResourceExhausted,
    GoogleAPICallError,
    ServiceUnavailable,
)
from google.generativeai.types import GenerationConfig
from pydantic import ValidationError

from src.models import CodeReviewResult, Issue, ReviewSession

if TYPE_CHECKING:
    from src.diff_parser import ChangedFile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert software engineer performing a thorough code review.
Your task is to analyze the provided code diff and identify issues.

For each file, you MUST return a JSON object that matches this schema exactly:
{
  "file_name": "<relative file path>",
  "issues": [
    {
      "line_number": <integer, the new-file line where the issue occurs>,
      "severity": "<critical | warning | info>",
      "issue_type": "<bug | security | style | performance>",
      "description": "<clear explanation of the issue>",
      "suggestion": "<actionable fix or improvement>"
    }
  ],
  "overall_quality_score": <integer 1-10>,
  "summary": "<high-level summary of the review>"
}

Guidelines:
- "critical": bugs that cause crashes, data loss, or security vulnerabilities.
- "warning": potential problems, poor practices, or logic errors.
- "info": style, readability, or minor improvements.
- overall_quality_score: 1 = terrible, 10 = production-ready, no issues.
- Focus ONLY on the changed lines shown in the diff (lines starting with '+').
- Do NOT invent issues not visible in the diff.
- Be concise but precise in descriptions and suggestions.
- If no issues are found, return an empty issues array with a high score.
- Return ONLY the JSON object. No prose, no markdown fences.
"""

USER_PROMPT_TEMPLATE = """\
Please review the following code diff for the file `{file_name}`.

```diff
{diff_text}
```

Return ONLY the JSON object described in the instructions. No prose, no markdown fences.
"""

# ---------------------------------------------------------------------------
# Generation config — enforces JSON output and bounds resource consumption
# (fixes CWE-400 LLM Unbounded Consumption)
# ---------------------------------------------------------------------------

_GENERATION_CONFIG = GenerationConfig(
    temperature=0.1,          # low temperature for deterministic structured output
    max_output_tokens=4096,   # hard cap — prevents unbounded token consumption
    response_mime_type="application/json",  # Gemini native JSON mode
)

# Request-level timeout in seconds passed via request_options
_REQUEST_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Reviewer class
# ---------------------------------------------------------------------------


class AIReviewer:
    """
    Orchestrates AI-powered code review using the Google Gemini API.

    Uses Gemini's native JSON response mode (response_mime_type) combined
    with full Pydantic validation and exponential-backoff retries.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0  # seconds

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=model or self.DEFAULT_MODEL,
            generation_config=_GENERATION_CONFIG,
            system_instruction=SYSTEM_PROMPT,
        )
        self.model_name = model or self.DEFAULT_MODEL
        self.max_retries = max_retries
        logger.debug("AIReviewer initialised with model=%s", self.model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review_file(self, changed_file: "ChangedFile") -> CodeReviewResult:
        """
        Review a single changed file and return a CodeReviewResult.

        Retries on transient API errors and Pydantic validation failures.
        """
        logger.info("Reviewing %s …", changed_file.file_path)

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw_json = self._call_gemini(changed_file)
                result = self._parse_response(raw_json, changed_file.file_path)
                logger.info(
                    "Review complete for %s — score=%d, issues=%d",
                    result.file_name,
                    result.overall_quality_score,
                    len(result.issues),
                )
                return result

            except ValidationError as exc:
                logger.warning(
                    "Attempt %d/%d — Pydantic validation failed for %s: %s",
                    attempt,
                    self.max_retries,
                    changed_file.file_path,
                    exc,
                )
                last_exc = exc

            except (ResourceExhausted, DeadlineExceeded, ServiceUnavailable) as exc:
                # ResourceExhausted  → quota / rate-limit  (analogous to RateLimitError)
                # DeadlineExceeded   → request timeout      (analogous to APITimeoutError)
                # ServiceUnavailable → transient 503        (safe to retry)
                delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Attempt %d/%d — Gemini API error (%s), retrying in %.1fs …",
                    attempt,
                    self.max_retries,
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)
                last_exc = exc

            except GoogleAPICallError as exc:
                # Non-transient Gemini error (e.g. invalid API key, bad request)
                logger.error(
                    "Non-retryable Gemini API error for %s: %s",
                    changed_file.file_path,
                    exc,
                )
                raise

        logger.error(
            "All %d attempts failed for %s. Returning fallback result.",
            self.max_retries,
            changed_file.file_path,
        )
        return self._fallback_result(changed_file.file_path, str(last_exc))

    def review_session(self, changed_files: list["ChangedFile"]) -> ReviewSession:
        """
        Review a list of changed files and return a ReviewSession.
        Files that are not reviewable are silently skipped.
        """
        results: list[CodeReviewResult] = []
        for cf in changed_files:
            if not cf.is_reviewable:
                logger.debug("Skipping non-reviewable file: %s", cf.file_path)
                continue
            result = self.review_file(cf)
            results.append(result)

        session = ReviewSession(results=results)
        logger.info(
            "Session complete — %d file(s) reviewed, avg score=%.1f, total issues=%d",
            len(results),
            session.average_quality_score,
            session.total_issues,
        )
        return session

    # ------------------------------------------------------------------
    # Gemini interaction
    # ------------------------------------------------------------------

    def _call_gemini(self, changed_file: "ChangedFile") -> str:
        """Call the Gemini API and return the raw JSON string."""
        user_prompt = USER_PROMPT_TEMPLATE.format(
            file_name=changed_file.file_path,
            diff_text=self._truncate_diff(changed_file.diff_text),
        )

        response = self._model.generate_content(
            user_prompt,
            request_options={"timeout": _REQUEST_TIMEOUT},
        )

        # Gemini can return multiple candidates; take the first non-empty one
        raw = ""
        if response.candidates:
            raw = response.candidates[0].content.parts[0].text or ""

        # Fallback to the convenience .text accessor
        if not raw:
            try:
                raw = response.text or ""
            except ValueError:
                # response.text raises ValueError when the response was blocked
                finish_reason = (
                    response.candidates[0].finish_reason
                    if response.candidates
                    else "unknown"
                )
                raise GoogleAPICallError(  # type: ignore[arg-type]
                    f"Gemini response was blocked or empty (finish_reason={finish_reason})"
                )

        logger.debug(
            "Raw Gemini response for %s:\n%s",
            changed_file.file_path,
            raw[:500],
        )
        return raw

    # ------------------------------------------------------------------
    # Parsing & validation  (provider-agnostic — identical to before)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw_json: str, file_path: str) -> CodeReviewResult:
        """
        Parse and validate the raw JSON string from Gemini.

        Raises pydantic.ValidationError if the schema doesn't match,
        which triggers a retry in review_file().
        """
        # Strip accidental markdown fences that some model versions emit
        # even when response_mime_type is set to application/json
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]  # drop opening fence line
            cleaned = cleaned.rsplit("```", 1)[0]  # drop closing fence
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            # Re-raise as a ValidationError so the retry loop handles it uniformly
            raise ValidationError.from_exception_data(  # type: ignore[call-arg]
                title="JSON decode error",
                input_type="python",
                line_errors=[],
            ) from exc

        # Ensure file_name is set from what we know, not what the model guesses
        data.setdefault("file_name", file_path)

        # Coerce line_number to int if the model returned a string
        for issue in data.get("issues", []):
            if isinstance(issue.get("line_number"), str):
                try:
                    issue["line_number"] = int(issue["line_number"])
                except ValueError:
                    issue["line_number"] = 1

        return CodeReviewResult.model_validate(data)

    @staticmethod
    def _fallback_result(file_path: str, error_msg: str) -> CodeReviewResult:
        """Return a minimal valid CodeReviewResult when all retries fail."""
        return CodeReviewResult(
            file_name=file_path,
            issues=[
                Issue(
                    line_number=1,
                    severity="info",
                    issue_type="bug",
                    description=f"AI review could not be completed: {error_msg[:200]}",
                    suggestion="Please review this file manually.",
                )
            ],
            overall_quality_score=5,
            summary="Automated review failed. Manual review required.",
        )

    @staticmethod
    def _truncate_diff(diff_text: str, max_chars: int = 12_000) -> str:
        """
        Truncate very large diffs to stay within the model's context window.
        Appends a notice so the model knows the diff was cut.
        """
        if len(diff_text) <= max_chars:
            return diff_text
        truncated = diff_text[:max_chars]
        # Try to cut at a clean line boundary
        last_newline = truncated.rfind("\n")
        if last_newline > max_chars // 2:
            truncated = truncated[:last_newline]
        truncated += "\n\n[... diff truncated due to size — review partial diff only ...]"
        logger.warning("Diff truncated from %d to %d chars", len(diff_text), len(truncated))
        return truncated
