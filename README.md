# Automated Code Quality Assessment via LLM Inference

Uses TinyLlama-1.1B to review code snippets for bugs, security holes, performance issues, and style problems. The model reads your code, thinks about what could go wrong, and gives you a structured report with severity ratings.

## how it works

```
code snippet → tokenizer → TinyLlama-1.1B → structured review
```

The system wraps TinyLlama in a chat-style prompt that instructs it to act like a senior code reviewer. It supports four review modes (general, security, performance, style) and can stream tokens back in real-time via SSE.

**model details:**
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B parameters)
- KV cache enabled for faster autoregressive decoding
- repetition penalty of 1.15 to reduce loops
- runs on CUDA if available, falls back to CPU

## setup

```bash
pip install -r requirements.txt
python main.py
```

server starts at `localhost:8001`. hit `/docs` for the swagger UI.

## api

| endpoint | method | what it does |
|----------|--------|-------------|
| `/health` | GET | status check, shows device info |
| `/review` | POST | full code review (blocking) |
| `/review/stream` | POST | token-by-token streaming via SSE |
| `/review/batch` | POST | review up to 5 snippets at once |

## architecture

```
core/
├── config.py      → model IDs, prompts, generation parameters
└── reviewer.py    → CodeReviewer class, inference + streaming

api/
├── schemas.py     → pydantic request/response models
└── routes.py      → endpoint handlers

main.py            → FastAPI entry point, lifespan
app.py             → streamlit frontend
```

## streamlit demo

```bash
streamlit run app.py
```

interactive UI where you paste code, pick a language and review focus, and get back a structured analysis.

## requirements

- python 3.10+
- ~3GB disk for model weights (downloaded on first run)
- GPU optional but recommended
