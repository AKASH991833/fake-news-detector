# Python/NLP version (scikit-learn)

The Pages demo runs a compact model fully in your browser. This folder is the
proper training pipeline with a larger vocabulary, character-aware options,
and full holdout + cross-validation reporting.

```bash
pip install -r requirements.txt
python pipeline.py [--data corpus.jsonl]
```

`corpus.jsonl` ships a small curated starter corpus (`{"text","label"}` per
line). Point `--data` at a larger open JSONL dataset (same schema) for
serious training. Metrics land in `metrics.json`; the model in `model.joblib`.

Honesty note (same as the demo): this is a text-pattern model — it measures
writing style, not truth. Always verify claims against primary sources.
