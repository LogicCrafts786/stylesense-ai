# StyleSense AI – Detailed Setup Guide

This guide walks through a complete local setup, from a clean machine to
a running app, plus common troubleshooting steps.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Check with `python --version` |
| pip | Comes with Python; upgrade with `python -m pip install --upgrade pip` |
| Google Gemini API key | Get one free at https://ai.google.dev/ |
| (Optional) OpenWeatherMap API key | Only needed for live weather-aware reasoning |
| Git | For cloning the repo |

---

## 2. Clone and Set Up a Virtual Environment

```bash
git clone https://github.com/<your-username>/stylesense-ai.git
cd stylesense-ai

python -m venv venv

# Activate:
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows (cmd)
venv\Scripts\Activate.ps1       # Windows (PowerShell)
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you plan to run tests or contribute code, also install dev dependencies:

```bash
pip install -e ".[dev]"
```

---

## 4. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```env
GEMINI_API_KEY=your_actual_key_here
```

All other variables have sensible defaults for local development. See
`.env.example` for the full list and inline comments.

### Getting a Gemini API Key
1. Go to https://ai.google.dev/
2. Sign in with a Google account.
3. Navigate to "Get API Key" and create a new key.
4. Paste it into `.env` as `GEMINI_API_KEY`.

### Getting a Weather API Key (Optional)
1. Go to https://openweathermap.org/api
2. Create a free account and generate an API key.
3. Paste it into `.env` as `WEATHER_API_KEY`.
4. If omitted, the app still works — it simply skips live weather lookups
   and relies on user-stated season/weather context instead.

---

## 5. Run the Application

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`. If it doesn't,
open that URL manually in your browser.

---

## 6. Verify the Setup

Try these sample interactions once the app loads:

1. **Text-only:** "I need an outfit for a business meeting, budget $150."
2. **Image upload:** Upload any clothing photo and ask "What goes well with this?"
3. **Comparison:** "Compare the Chelsea boots and the leather sneakers."
4. **Review summary:** "What do people say about the wrap dress?"

If all four return sensible responses, your setup is working correctly.

---

## 7. Running Tests

```bash
pytest                          # full suite with coverage
pytest -m unit                  # unit tests only (no live API calls)
pytest tests/test_tools/ -v     # a specific test directory, verbose
```

---

## 8. Troubleshooting

| Issue | Likely Cause | Fix |
|---|---|---|
| `ConfigurationError: GEMINI_API_KEY is not set` | `.env` missing or key not set | Copy `.env.example` → `.env`, add key, restart |
| `ModuleNotFoundError: No module named 'src'` | Running from wrong directory, or package not installed | Run `streamlit run app.py` from repo root, or `pip install -e .` |
| Gemini API errors / rate limits | Free tier quota exceeded | Wait and retry, or check quota at https://ai.google.dev/ |
| ChromaDB fails to initialize | Corrupted persistence directory | Delete `chroma_db/` folder and restart (it will rebuild) |
| Image upload rejected | File too large or unsupported format | Use JPEG/PNG/WEBP under `MAX_UPLOAD_SIZE_MB` (default 10MB) |
| Weather data always unavailable | `WEATHER_API_KEY` not set | This is expected/optional; app falls back to season-hint reasoning |

---

## 9. Switching Vector Store Backends

By default the app uses ChromaDB. To switch to FAISS:

```env
VECTOR_STORE_TYPE=faiss
```

No other code changes are needed — `src/memory/vector_store.py` selects
the backend automatically based on this setting.

---

## 10. Production Deployment Notes

This repo is structured for local/demo use out of the box. For production:

- Replace file-based `UserProfileMemory` storage with a real database.
- Replace in-process `ConversationMemoryStore` with Redis or similar for
  multi-instance deployments.
- Set `APP_ENV=production` and review `LOG_LEVEL` (default `INFO`).
- Consider enabling live product scraping only behind explicit user consent
  and rate-limit safeguards (see `scraper_service.py`).
