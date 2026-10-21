# Fake News Detector — Spot Misinformation with On-Device ML

**Live:** https://akash991833.github.io/fake-news-detector/

A fake-news and clickbait style-checker that shows its work: a logistic-regression model trained on 123 hand-labeled headlines scores any headline and highlights the exact words that smell like misinformation. Ships pre-trained; retrain live in your browser. No server, no uploads, no sign-up.

![Desktop](screenshots/desktop-light.png)

## Features

- **Instant verdicts** — Likely fake / Likely real / Unclear, with a fake-news score
- **Word-level explanations** — occlusion highlights show which words push toward fake (red) or real (green)
- **Batch screening** — paste a feed of headlines, get a summary and table, export as CSV
- **Live retraining** — retrain in your browser any time; nothing leaves your device
- **Honest metrics** — real 5-fold cross-validation with a confusion matrix
- **PWA** — installable, works fully offline
- **Dark & light themes**, responsive on mobile and desktop

## Honest metrics (5-fold cross-validation, 123 headlines)

| Metric | Score |
|---|---|
| Accuracy | 88.6% |
| Precision | 91.4% |
| Recall | 85.5% |
| F1 | 0.88 |

## Honest limits — read this

This model checks **writing style, not facts**. It learned that fake news often uses clickbait patterns (SHOCKING, one weird trick, they don't want you to know) while real news uses sober, specific language. It cannot verify whether a claim is true: a well-written lie will pass, and a sensational-but-true headline will be flagged. It is a style signal to prompt a second look — always verify important claims with trusted sources. English only; trained on 123 hand-written headlines.

## How it works

1. **Tokenize + stem + negation merge** — "did not happen" → `not_happen`
2. **Bag-of-words** — each headline becomes a binary vector over a 180-word vocabulary
3. **Logistic regression + L2** — one learned weight per word (TensorFlow.js)
4. **Occlusion explanations** — each word is removed and the score change measured

## Tech

Vanilla HTML/CSS/JS + TensorFlow.js (CDN). No build step, no backend, no tracking.

## Run locally

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

Built by [Akash Vishwakarma](https://github.com/AKASH991833). Free to use — no sign-up, no card.


## Stacks

- **Live demo (this repo root):** HTML/CSS/JS — on-device text-pattern model with disclosed accuracy.
- **Python pipeline (`ml/`):** scikit-learn TF-IDF + LogisticRegression with full holdout + 5-fold CV metrics, trainable on larger JSONL corpora. See `ml/README.md`.
