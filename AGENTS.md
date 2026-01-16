# Repository Guidelines

## Project Structure & Module Organization
- `reference.py` is the main GTK/Adwaita application entry point.
- `config.json` stores local runtime settings (API endpoints, keys, prompt text).
- `prior_briefing_odt_files/` contains source ODT briefs that are ingested.
- `reference_data/` holds the SQLite index (`briefs.sqlite`) and Chroma data.
- `prompts/` contains prompt text used during development.

## Build, Test, and Development Commands
- `python3 reference.py` runs the desktop app.
- `uv sync` installs dependencies from `pyproject.toml`/`uv.lock` (if using uv).
- `uv add <package>` updates dependencies when adding libraries (preferred by this project).

## Coding Style & Naming Conventions
- Python 3.13+ only (see `pyproject.toml`).
- Indentation: 4 spaces; follow standard Python typing conventions.
- Constants are uppercase (e.g., `CONFIG_FILE`), functions are `snake_case`, classes are `PascalCase`.
- Keep UI logic in `reference.py` and avoid mixing in sample files unless explicitly promoting a pattern.

## Testing Guidelines
- No automated test suite is present. If you add tests, document how to run them (e.g., `pytest`) and keep fixtures outside `reference_data/`.

## Commit & Pull Request Guidelines
- This directory is not a standalone git repo, so there is no commit history to follow.
- Use clear, imperative commit subjects (e.g., "Add RAG settings validation").
- PRs should describe user-facing behavior changes and list any config or data migrations.

## Security & Configuration Tips
- Treat `config.json` as local-only; do not commit real API keys.
- If you need to share settings, provide a redacted example or a template file.
- Large data in `prior_briefing_odt_files/` and `reference_data/` should be updated intentionally and avoided in diffs unless required.
