# 🤖 AI Code Review CLI

An AI-powered code review tool that integrates Google Gemini with GitHub pull requests.  
It analyses your code diffs, returns structured review results, and posts inline comments directly on your PRs.

---

## Features

- **Structured AI output** — uses Gemini's native `application/json` response mode and Pydantic v2 to guarantee well-typed results
- **Three review modes** — GitHub PR review, local file review, local git diff review
- **Inline PR comments** — posts per-issue comments at the exact line numbers in the diff
- **Rich terminal output** — coloured severity tables and score panels when running locally
- **GitHub Actions integration** — auto-reviews every PR on `opened` / `synchronize`
- **Retry logic** — exponential backoff on rate-limit and timeout errors, with graceful Pydantic validation retries

---

## Project Structure

```
.
├── src/
│   ├── __init__.py          # Package init
│   ├── cli.py               # Typer entry point & Rich display
│   ├── models.py            # Pydantic v2 schemas (Issue, CodeReviewResult, ReviewSession)
│   ├── reviewer.py          # Gemini API calls + retry logic
│   ├── github_client.py     # PyGithub integration (fetch diffs, post comments)
│   └── diff_parser.py       # GitPython diff parsing + local file wrapper
├── .github/
│   └── workflows/
│       └── code-review.yml  # GitHub Actions workflow
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- A Google Gemini API key (get one free at [Google AI Studio](https://aistudio.google.com/app/apikey))
- A GitHub personal access token (or rely on `GITHUB_TOKEN` in Actions)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/ai-code-review-cli.git
cd ai-code-review-cli

# Create and activate a virtual environment
python -m venv .venv
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment variables

```bash
export GEMINI_API_KEY="AIza..."
export GITHUB_TOKEN="ghp_..."    # required only for PR review mode
```

On Windows (PowerShell):

```powershell
$env:GEMINI_API_KEY = "AIza..."
$env:GITHUB_TOKEN   = "ghp_..."
```

---

## Usage

### Review a GitHub Pull Request

Fetches the PR diff, runs AI review, and posts inline comments back to GitHub.

```bash
python -m src.cli --pr-number 42 --repo your-org/your-repo
```

Use `--no-post` to do a dry-run (review without posting to GitHub):

```bash
python -m src.cli --pr-number 42 --repo your-org/your-repo --no-post
```

### Review a Local File

```bash
python -m src.cli --file src/auth/login.py
```

### Review Local Git Diff

Compares your current branch against `main` (configurable via `--base`):

```bash
python -m src.cli --diff
python -m src.cli --diff --base develop
```

### Full Options

```
Options:
  -p, --pr-number INTEGER   GitHub Pull Request number
  -r, --repo TEXT           GitHub repository (owner/repo)
  -f, --file PATH           Local file to review
  -d, --diff                Review local git diff vs base branch
  -b, --base TEXT           Base branch for diff [default: main]
  -m, --model TEXT          Gemini model [default: gemini-2.5-flash]
  -v, --verbose             Enable debug logging
      --no-post             Skip posting results to GitHub
      --help                Show this message and exit
```

---

## Data Models

### `Issue`

| Field         | Type                                          | Description                              |
|---------------|-----------------------------------------------|------------------------------------------|
| `line_number` | `int` (≥ 1)                                   | Line in the file where the issue occurs  |
| `severity`    | `"critical"` \| `"warning"` \| `"info"`       | Severity level                           |
| `issue_type`  | `"bug"` \| `"security"` \| `"style"` \| `"performance"` | Category of the issue         |
| `description` | `str`                                         | Explanation of the problem               |
| `suggestion`  | `str`                                         | Actionable fix or improvement            |

### `CodeReviewResult`

| Field                   | Type          | Description                            |
|-------------------------|---------------|----------------------------------------|
| `file_name`             | `str`         | Relative file path                     |
| `issues`                | `List[Issue]` | All issues found                       |
| `overall_quality_score` | `int` (1–10)  | Quality score for this file            |
| `summary`               | `str`         | High-level review summary              |

---

## GitHub Actions Setup

1. Add your `GEMINI_API_KEY` as a repository secret:  
   **Settings → Secrets and variables → Actions → New repository secret**

2. `GITHUB_TOKEN` is provided automatically by GitHub Actions — no extra setup needed.

3. The workflow file at `.github/workflows/code-review.yml` will trigger automatically on every PR.

---

## Severity Guide

| Level      | Colour | Meaning                                                    |
|------------|--------|------------------------------------------------------------|
| `critical` | 🔴 Red  | Crashes, data loss, security vulnerabilities — must fix   |
| `warning`  | 🟡 Yellow | Potential bugs, poor practices — should fix              |
| `info`     | 🔵 Blue  | Style, readability, minor improvements — nice to fix      |

---

## Quality Score Guide

| Score | Colour  | Meaning                                  |
|-------|---------|------------------------------------------|
| 1–3   | 🔴 Red   | Poor — significant issues present        |
| 4–6   | 🟡 Yellow | Average — several improvements needed   |
| 7–10  | 🟢 Green | Good to excellent — production ready    |

---

## Extending the Tool

- **Different AI provider**: Swap `src/reviewer.py` to use the OpenAI or Anthropic client — the Pydantic models and CLI are provider-agnostic.
- **More file types**: Add extensions to `REVIEWABLE_EXTENSIONS` in `src/diff_parser.py`.
- **Custom prompts**: Edit `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE` in `src/reviewer.py`.
- **Slack/Teams notifications**: Add a notifier module and call it from `_run_review_and_display` in `src/cli.py`.

---

## License

MIT — see [LICENSE](LICENSE) for details.
