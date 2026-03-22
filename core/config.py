from dataclasses import dataclass, field


@dataclass
class ReviewConfig:
    """all the knobs for the code reviewer in one place."""

    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_input_tokens: int = 2048
    default_max_output: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    use_fp16: bool = True       # only kicks in when cuda is available
    port: int = 8001


# ---- prompt templates ----

SYSTEM_PROMPT = (
    "You are an expert code reviewer. Analyze the code below and provide:\n"
    "1. **Bugs & Issues**: Any logical errors, potential crashes, or incorrect behavior\n"
    "2. **Security**: SQL injection, XSS, command injection, hardcoded secrets, etc.\n"
    "3. **Performance**: Inefficient algorithms, unnecessary allocations, N+1 queries\n"
    "4. **Style**: Naming conventions, code organization, readability improvements\n"
    "5. **Summary**: One-line verdict (APPROVE / REQUEST_CHANGES / NEEDS_DISCUSSION)\n\n"
    "Be specific. Reference line numbers. Be concise but thorough."
)

FOCUS_PROMPTS = {
    "general": "Review all aspects of this code.",
    "security": "Focus primarily on security vulnerabilities and data safety.",
    "performance": "Focus primarily on performance bottlenecks and optimization.",
    "style": "Focus primarily on code style, readability, and best practices.",
}

SUPPORTED_LANGUAGES = ["python", "javascript", "java", "go", "rust", "c", "cpp"]
