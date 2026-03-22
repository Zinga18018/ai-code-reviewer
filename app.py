"""
streamlit frontend for the code reviewer.
paste code, pick a focus area, get a review.
"""

import streamlit as st
from core import CodeReviewer, ReviewConfig
from core.config import SUPPORTED_LANGUAGES

st.set_page_config(page_title="Code Reviewer", layout="wide")


@st.cache_resource
def load_model():
    reviewer = CodeReviewer(ReviewConfig())
    reviewer.load()
    return reviewer


st.title("Code Quality Assessment")
st.caption("powered by TinyLlama-1.1B")

reviewer = load_model()

col1, col2 = st.columns(2)

with col1:
    code = st.text_area(
        "paste your code here", height=400,
        placeholder="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    )

    language = st.selectbox("language", SUPPORTED_LANGUAGES)
    focus = st.selectbox("review focus", ["general", "security", "performance", "style"])
    max_tokens = st.slider("max output tokens", 128, 512, 300)

    run = st.button("review code", type="primary", use_container_width=True)

with col2:
    if run and code.strip():
        with st.spinner("running inference..."):
            result = reviewer.review(code, language=language, focus=focus, max_tokens=max_tokens)

        st.subheader("review")
        st.markdown(result["review"])

        st.divider()
        cols = st.columns(3)
        cols[0].metric("model", "TinyLlama-1.1B")
        cols[1].metric("inference", f"{result['inference_ms']:.0f}ms")
        cols[2].metric("device", result["device"])

    elif run:
        st.warning("paste some code first")
