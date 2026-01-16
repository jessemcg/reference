# Reference

![Reference app icon](reference.png)

Desktop GTK/Adwaita app that ingests prior briefing ODT files and lets you ask
legal questions with searchable results you can paste into LibreOffice Writer.

## Requirements

- Python 3.13+
- GTK/Adwaita (via `pygobject`)
- Optional: `uv` for dependency management

## Quick start

1. Install dependencies:
   - `uv sync`
2. Run the app:
   - `python3 reference.py`

## Configuration

- `config.json` stores local runtime settings (API endpoints, keys, prompt text).
- Treat `config.json` as local-only; do not commit real API keys.

## Project layout

- `reference.py`: main application entry point.
- `prior_briefing_odt_files/`: source ODT briefs that are ingested.
- `reference_data/`: SQLite index (`briefs.sqlite`) and Chroma data.
- `prompts/`: prompt text used during development.

## Notes

- This directory is not a standalone git repository.
- No automated tests are currently defined.
