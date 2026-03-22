from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from .schemas import ReviewRequest, ReviewResponse


def register_routes(app, reviewer):
    """wire up all API endpoints."""

    @app.get("/health")
    async def health():
        return reviewer.health()

    @app.post("/review", response_model=ReviewResponse)
    async def review_code(req: ReviewRequest):
        if not reviewer.is_loaded:
            raise HTTPException(503, "model is still loading")

        result = reviewer.review(req.code, req.language, req.focus, req.max_tokens)
        return ReviewResponse(
            review=result["review"],
            language=result["language"],
            focus=result["focus"],
            inference_ms=result["inference_ms"],
            model=result["model"],
            device=result["device"],
        )

    @app.post("/review/stream")
    async def review_stream(req: ReviewRequest):
        if not reviewer.is_loaded:
            raise HTTPException(503, "model is still loading")

        async def event_generator():
            for token in reviewer.stream(req.code, req.language, req.focus, req.max_tokens):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/review/batch")
    async def review_batch(requests: list[ReviewRequest]):
        if not reviewer.is_loaded:
            raise HTTPException(503, "model is still loading")

        if len(requests) > 5:
            raise HTTPException(400, "max 5 reviews per batch")

        results = []
        for req in requests:
            result = reviewer.review(req.code, req.language, req.focus, req.max_tokens)
            results.append(ReviewResponse(
                review=result["review"],
                language=result["language"],
                focus=result["focus"],
                inference_ms=result["inference_ms"],
                model=result["model"],
                device=result["device"],
            ))
        return results
