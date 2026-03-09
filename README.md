# Reference

<img src="reference.png" alt="Reference" width="128" align="left">

Desktop GTK/Adwaita app for building a local research index from prior briefing
ODT files and querying it through search and RAG workflows.

Reference combines:

- full-text brief search
- local SQLite and Chroma-backed indexing
- multiple RAG prompt modes
- configurable Voyage and Isaacus model settings
- output formatted for easy reuse in LibreOffice Writer

<br clear="left">

## Requirements

- Python 3.13+
- GTK/Adwaita (via `pygobject`)
- Pandoc available on `PATH`
- Optional: `uv` for dependency management

## Quick start

1. Install dependencies with `uv sync`.
2. Make sure local runtime settings exist in `config.json`.
3. Run `python3 reference.py`.

## Configuration

- `config.json` stores local runtime settings (API endpoints, keys, prompt text).
- `config.json` is ignored locally and should never be committed with real API keys.
- The app supports provider, model, prompt, timeout, and output display settings.

## Project layout

- `reference.py`: main application entry point.
- `prior_briefing_odt_files/`: source ODT briefs that are ingested.
- `reference_data/`: SQLite index (`briefs.sqlite`) and Chroma data.
- `prompts/`: prompt text used during development.

## Notes

- No automated tests are currently defined.
