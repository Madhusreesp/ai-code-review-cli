"""Tests for src/diff_parser.py — ChangedFile and DiffParser helpers."""

import pytest

from src.diff_parser import (
    REVIEWABLE_EXTENSIONS,
    ChangedFile,
    DiffParser,
    parse_local_file,
)


# ---------------------------------------------------------------------------
# ChangedFile
# ---------------------------------------------------------------------------

def _make_changed_file(**overrides) -> ChangedFile:
    base = dict(
        file_path="src/app.py",
        old_content="x = 1\n",
        new_content="x = 2\n",
        diff_text="--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n",
        changed_line_numbers=[1],
    )
    base.update(overrides)
    return ChangedFile(**base)


class TestChangedFile:
    def test_extension_extracted(self):
        cf = _make_changed_file(file_path="src/utils.py")
        assert cf.extension == ".py"

    def test_is_reviewable_true_for_py(self):
        cf = _make_changed_file(file_path="main.py")
        assert cf.is_reviewable is True

    def test_is_reviewable_false_for_txt(self):
        cf = _make_changed_file(file_path="notes.txt")
        assert cf.is_reviewable is False

    def test_is_reviewable_false_for_md(self):
        cf = _make_changed_file(file_path="README.md")
        assert cf.is_reviewable is False

    @pytest.mark.parametrize("ext", [".js", ".ts", ".go", ".java", ".rs", ".rb"])
    def test_is_reviewable_for_common_extensions(self, ext):
        cf = _make_changed_file(file_path=f"file{ext}")
        assert cf.is_reviewable is True

    def test_change_summary_counts_additions(self):
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1,2 @@\n-old\n+new1\n+new2\n"
        cf = _make_changed_file(diff_text=diff)
        summary = cf.change_summary
        assert "+2" in summary
        assert "-1" in summary

    def test_extension_case_insensitive(self):
        cf = _make_changed_file(file_path="Script.PY")
        assert cf.extension == ".py"
        assert cf.is_reviewable is True


# ---------------------------------------------------------------------------
# DiffParser._extract_changed_line_numbers
# ---------------------------------------------------------------------------

class TestExtractChangedLineNumbers:
    def _extract(self, diff_text: str) -> list[int]:
        return list(DiffParser._extract_changed_line_numbers(diff_text))

    def test_single_hunk(self):
        # Hunk starts at new-file line 1.
        # Line 1: context  (+1 → current_line=2)
        # Line 2: -old line (skipped, no increment)
        # Line 3: +new line → yields current_line=2, then increments to 3
        diff = "@@ -1,3 +1,3 @@\n context\n-old line\n+new line\n context\n"
        lines = self._extract(diff)
        assert 2 in lines  # '+new line' lands at new-file line 2

    def test_addition_only_hunk(self):
        diff = "@@ -0,0 +1,2 @@\n+line one\n+line two\n"
        lines = self._extract(diff)
        assert lines == [1, 2]

    def test_empty_diff_returns_empty(self):
        assert self._extract("") == []

    def test_multiple_hunks(self):
        diff = (
            "@@ -1,2 +1,2 @@\n-a\n+A\n context\n"
            "@@ -10,2 +10,2 @@\n-b\n+B\n context\n"
        )
        lines = self._extract(diff)
        assert 1 in lines
        assert 10 in lines

    def test_header_lines_not_counted(self):
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n"
        lines = self._extract(diff)
        # Only the '+new' line should be counted, not the +++ header
        assert lines == [1]


# ---------------------------------------------------------------------------
# DiffParser.parse_diff_text
# ---------------------------------------------------------------------------

class TestParseDiffText:
    def test_returns_changed_file(self):
        diff = "@@ -1 +1 @@\n-old\n+new\n"
        parser = DiffParser()
        cf = parser.parse_diff_text(diff, "src/foo.py")
        assert cf.file_path == "src/foo.py"
        assert cf.diff_text == diff
        assert isinstance(cf.changed_line_numbers, list)

    def test_unknown_default_path(self):
        parser = DiffParser()
        cf = parser.parse_diff_text("@@ -1 +1 @@\n+x\n")
        assert cf.file_path == "unknown"


# ---------------------------------------------------------------------------
# parse_local_file
# ---------------------------------------------------------------------------

class TestParseLocalFile:
    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_local_file("nonexistent_file_xyz.py")

    def test_wraps_file_as_changed_file(self, tmp_path):
        f = tmp_path / "sample.py"
        f.write_text("x = 1\ny = 2\n", encoding="utf-8")
        cf = parse_local_file(str(f))
        assert cf.file_path == str(f)
        assert cf.new_content == "x = 1\ny = 2\n"
        assert cf.changed_line_numbers == [1, 2]

    def test_synthetic_diff_has_additions(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("a = 1\n", encoding="utf-8")
        cf = parse_local_file(str(f))
        added = [l for l in cf.diff_text.splitlines() if l.startswith("+") and not l.startswith("+++")]
        assert len(added) == 1
        assert "a = 1" in added[0]

    def test_is_reviewable_for_py(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("pass\n", encoding="utf-8")
        cf = parse_local_file(str(f))
        assert cf.is_reviewable is True
