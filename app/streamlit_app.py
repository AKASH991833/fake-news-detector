"""Fake News Detector — Streamlit UI over the trained TF-IDF + LogReg pipeline."""
import json
from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
pipe = joblib.load(ROOT / "ml" / "model.joblib")
metrics = json.load(open(ROOT / "ml" / "metrics.json"))

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.caption("TF-IDF + logistic regression, served live · [Browser demo](https://akash991833.github.io/fake-news-detector/) · [Code](https://github.com/AKASH991833/fake-news-detector)")

text = st.text_area("Paste an article, headline or claim", height=180,
                    placeholder="BREAKING: Scientists confirm ...")
if st.button("Analyze", type="primary") and text.strip():
    proba = pipe.predict_proba([text])[0]
    classes = list(pipe.classes_)
    p_fake = float(proba[classes.index(1) if 1 in classes else -1])
    if p_fake >= 0.5:
        st.error(f"⚠️ Patterns associated with fake news — score **{p_fake:.0%}**")
    else:
        st.success(f"Patterns associated with reliable writing — fake score **{p_fake:.0%}**")
    st.progress(p_fake)

with st.expander("Model & honest metrics"):
    st.json(metrics)
    st.warning(
        "This model reads **writing style patterns, not truth**. A well-written fake can pass; "
        "an unusual genuine piece can be flagged. Always verify claims against primary sources."
    )
