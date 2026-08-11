# LLM Gateway

A FastAPI service that sits between callers and multiple LLM providers (Groq and Gemini, via
LangChain). It authenticates callers, routes to the right provider/model, normalizes
provider-specific response shapes, logs every request, enforces per-client rate limits, and fails
over to a backup provider on retryable errors.

## Structure

```
app/
  main.py              # FastAPI app factory, mounts routers
  config.py            # settings loaded from .env
  database.py          # SQLAlchemy engine/session
  models.py            # SQLAlchemy models (Client, RequestLog)
  schemas.py           # Pydantic request/response models
  dependencies.py      # get_db, API-key auth dependency
  routers/
    clients.py         # POST /clients
    chat.py             # POST /v1/chat/completions
  providers/
    registry.py         # model -> provider registry, fallback resolution
    errors.py            # retryable-error classification
    text.py               # response normalization across providers
  services/
    rate_limit.py         # sliding-window rate limiting

demo/                  # PowerShell scripts for manual testing (gitignored)
tests/                 # placeholder, no automated tests yet
```

## Setup

```powershell
python -m venv gatewayllmenv
gatewayllmenv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # fill in GROQ_API_KEY, GOOGLE_API_KEY, ADMIN_API_KEY
```

## Run

```powershell
uvicorn app.main:app --reload --port 8080
```

Port 8000 is blocked on some Windows setups (`WinError 10013`); 8080 is used by convention here.

## Known gaps

See `CLAUDE.md` for the full list of deliberately deferred work (migrations, Redis-backed rate
limiting, TPM limits, mid-stream fallback test coverage, etc.).
