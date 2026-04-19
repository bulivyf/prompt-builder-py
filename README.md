# LLM Interactive Prompt Builder (Local‑First, Web‑Based, Accessible)

A local web app that guides you through:
1) entering an initial prompt
2) answering LLM‑generated clarifying questions
3) generating a refined prompt
4) accepting to produce **two final prompt variants** (human-friendly + LLM-optimized)
5) saving to SQLite, saving to a file, and copying to clipboard

## Quick start (Windows 11)

1. **Create and activate venv**
   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bat
   pip install -r requirements.txt
   ```

3. **(Recommended) Configure .env**
   Copy `.env.example` to `.env` and edit values:
- Default provider is **local**.
- To use a remote provider, explicitly set `LLM_PROVIDER` (or legacy `LLM_Provider`) to one of: `openai`, `anthropic`, `google`, `grok`, `huggingface`.
- For *any other provider name*, define `PROVIDER_<NAME>_API_KEY`, `PROVIDER_<NAME>_BASE_URL`, and `PROVIDER_<NAME>_MODEL` (see `.env.example`).

4. **Run**
   ```bat
   scripts\run_dev.bat
   ```

5. **Open the webpage**
   - http://127.0.0.1:8012

## Local LLM setup (default path)

This app defaults to a local model when no external API keys exist.

1. Create a folder named `models/` at the project root.
2. Download a GGUF model (example: a small instruct model) and place it at:
   - `models\model.gguf` (or update `LOCAL_MODEL_PATH` in your `.env`)

> The local backend uses **llama-cpp-python**. If you hit an error about it, ensure it installed successfully, or reinstall with the right wheel for your Python version.

## SQLite database

A SQLite database file is created automatically at:
- `app_data\prompt_builder.sqlite3`

It stores:
- initial prompt
- clarifying questions (JSON)
- answers (JSON)
- refined prompt
- final prompt variants
- timestamp

## Notes on Accessibility (508 / WCAG A/AA)
- Semantic HTML (labels, fieldsets, legends)
- Keyboard navigation with visible focus states
- ARIA roles where needed
- High-contrast dark theme
- Screen-reader friendly status updates (aria-live regions)

## Project structure
- `app/main.py` – FastAPI server + endpoints
- `app/llm.py` – provider abstraction (local/remote)
- `app/db.py` / `app/models.py` – SQLite setup
- `app/templates/index.html` – UI
- `app/static/styles.css` / `app/static/app.js` – styling + interactions
- `scripts/run_dev.bat` – Windows launcher

## Security
This is intended for local use and binds to `127.0.0.1` by default.
