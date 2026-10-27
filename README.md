# AI Code Reviewer

TinyLlama-1.1B generates free-form suggestions about a code snippet, using general, security, performance or style prompts. It does not run the code, perform static analysis or validate its findings. The API returns review text, not a parsed report with severity ratings.

## Run locally

Use Python 3.10+ and a virtual environment. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8001
```

On Windows PowerShell, replace the activation line with `.\.venv\Scripts\Activate.ps1`. Open `http://127.0.0.1:8001/docs`.

Startup loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, downloading it if it is not cached. The code selects CUDA when available, otherwise CPU. Dependencies are unpinned; a working environment and GPU compatibility are not guaranteed by the requirements list.

The explicit [Uvicorn host setting](https://uvicorn.dev/settings/) keeps this launch on loopback. `python main.py` instead binds to `0.0.0.0` with reload enabled. The API has no authentication and permits all CORS origins; do not expose it as a public review service.

## API and interface

| Method | Path | Behavior |
|---|---|---|
| GET | `/health` | Model/device status |
| POST | `/review` | One generated review |
| POST | `/review/stream` | Generated tokens via SSE |
| POST | `/review/batch` | Up to five snippets, reviewed sequentially |

`streamlit run app.py` starts a separate interface for pasting code and choosing a review focus. It loads its own model instance rather than calling the API; running both consumes additional memory.

## Implementation and limits

[core/reviewer.py](core/reviewer.py) tokenizes the prompt, truncates input to 2,048 tokens and samples generated text. [core/config.py](core/config.py) sets temperature 0.7, repetition penalty 1.1 and KV caching. Responses are defined in [api/schemas.py](api/schemas.py).

Generated advice can miss vulnerabilities or invent problems. Treat it as a suggestion to inspect, not approval, a security audit or evidence that code is correct. No test suite, quality evaluation or timing benchmark is committed.

Setup, model execution and browser flows were not rerun for this documentation update.

## Related implementations

This project also appears in [ML-Portfolio-Projects](https://github.com/Zinga18018/ML-Portfolio-Projects), as an [API variant](https://github.com/Zinga18018/ML-Portfolio-Projects/tree/master/project1-ai-code-reviewer) and an [HF Space variant](https://github.com/Zinga18018/ML-Portfolio-Projects/tree/master/hf-spaces/ai-code-reviewer). Their code differs: the HF variant parses severity labels and computes a heuristic score; this repository returns free-form text. This README describes only this implementation.

