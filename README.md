# Jimmy — your personal AI assistant

Single-file web app (`jimmy.html`) + FastAPI backend (`jimmy_server.py`).
No Hugging Face, no paid API. The app runs on GitHub Pages, the backend on
Render's free tier, and the model is Google's Gemini free tier.

## Live

- App: https://jar5ar.github.io/jimmy/
- Backend: https://jimmy-backend.onrender.com (Render free tier)

Open the app, tap Settings, paste the backend address and your access
token, and chat. The token lives in the Render dashboard (Environment tab)
under `JIMMY_ACCESS_TOKEN` — copy it from there, never commit it here.

## How it is deployed

- `jimmy.html` is served by GitHub Pages from this repo (`main` branch, root).
- The backend is a Render Web Service created from this repo (Blueprint in
  `render.yaml`). Render installs `requirements.txt` and runs
  `uvicorn jimmy_server:app --host 0.0.0.0 --port $PORT`.
- The model is Gemini via its OpenAI-compatible endpoint. Get a free key at
  Google AI Studio (https://aistudio.google.com/apikey) and set it as the
  `MODEL_API_KEY` env var on Render. `MODEL_NAME` is `gemini-2.0-flash`.

## Environment variables (backend)

| Variable | Value |
|---|---|
| `JIMMY_ACCESS_TOKEN` | Long random secret (≥32 chars). Render generates one. |
| `MODEL_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `MODEL_API_KEY` | Your free Gemini API key from Google AI Studio. |
| `MODEL_NAME` | `gemini-2.0-flash` |
| `ALLOWED_ORIGIN` | `https://jar5ar.github.io` (the Pages site — CORS allows only this origin) |
| `JIMMY_DB_PATH` | Optional. Defaults to `jimmy_memory.sqlite3`. |

## Run it locally

```bash
pip install -r requirements.txt
export JIMMY_ACCESS_TOKEN=<long-random-secret>
export MODEL_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export MODEL_API_KEY=<your-gemini-key>
export MODEL_NAME=gemini-2.0-flash
export ALLOWED_ORIGIN=http://localhost:8000
uvicorn jimmy_server:app --host 0.0.0.0 --port 7860
```

Then open `jimmy.html` in a browser (or serve this folder) and point
Settings at `http://localhost:7860`.
