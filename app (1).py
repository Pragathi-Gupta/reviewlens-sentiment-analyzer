import re
import tempfile
from collections import Counter

import gradio as gr
import pandas as pd
from transformers import pipeline

# Multilingual model that predicts 1-5 stars, a natural fit for product reviews.
MODEL_ID = "nlptown/bert-base-multilingual-uncased-sentiment"
clf = pipeline("text-classification", model=MODEL_ID, top_k=None, truncation=True, max_length=256)

MAX_ROWS = 500  # keeps the free CPU Space responsive

STOPWORDS = set(
    """this that with have from they them were been very just than then also into only
    there their what when where which would could should about after before because
    really much more most some such even still will your you the and for are but not
    was had has its it's i'm i've don't didn't doesn't isn't wasn't product item order
    bought buy got get one two""".split()
)


def to_sentiment(stars: int) -> str:
    if stars <= 2:
        return "Negative"
    if stars == 3:
        return "Neutral"
    return "Positive"


def predict(texts):
    """Return a list of (stars, confidence, sentiment) for each text."""
    results = clf(list(texts), batch_size=16)
    out = []
    for scores in results:
        best = max(scores, key=lambda d: d["score"])
        stars = int(best["label"].split()[0])
        out.append((stars, best["score"], to_sentiment(stars)))
    return out


# ---------- Tab 1: single review ----------
def analyze_one(text):
    if not text or not text.strip():
        return {}, "Please type or paste a review."
    scores = clf([text])[0]
    p = {int(d["label"].split()[0]): d["score"] for d in scores}
    probs = {
        "Negative": p.get(1, 0) + p.get(2, 0),
        "Neutral": p.get(3, 0),
        "Positive": p.get(4, 0) + p.get(5, 0),
    }
    stars = max(p, key=p.get)
    verdict = f"**{to_sentiment(stars)}** - predicted rating: {'★' * stars}{'☆' * (5 - stars)} ({stars}/5)"
    return probs, verdict


# ---------- Tab 2: bulk CSV ----------
def find_text_column(df):
    for c in df.columns:
        if c.strip().lower() in ("review", "reviews", "text", "comment", "review_text"):
            return c
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    return obj_cols[0] if obj_cols else None


def find_product_column(df):
    for c in df.columns:
        if c.strip().lower() in ("product", "product_name", "item", "asin"):
            return c
    return None


def top_complaint_words(negative_texts, n=10):
    words = Counter()
    for t in negative_texts:
        for w in re.findall(r"[a-zA-Z']{4,}", str(t).lower()):
            if w not in STOPWORDS:
                words[w] += 1
    return words.most_common(n)


def analyze_csv(file):
    empty = ("Upload a CSV with a `review` column (and optionally a `product` column).",
             pd.DataFrame({"sentiment": [], "count": []}), pd.DataFrame(), pd.DataFrame(), None)
    if file is None:
        return empty
    try:
        df = pd.read_csv(file.name if hasattr(file, "name") else file)
    except Exception as e:
        return (f"Could not read the CSV: {e}",) + empty[1:]

    text_col = find_text_column(df)
    if text_col is None:
        return ("No text column found. Name your review column `review`.",) + empty[1:]

    df = df.dropna(subset=[text_col]).head(MAX_ROWS).reset_index(drop=True)
    if df.empty:
        return ("The CSV has no reviews.",) + empty[1:]

    preds = predict(df[text_col].astype(str))
    df["stars"] = [p[0] for p in preds]
    df["confidence"] = [round(p[1], 3) for p in preds]
    df["sentiment"] = [p[2] for p in preds]

    counts = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    chart = pd.DataFrame({"sentiment": counts.index, "count": counts.values})

    total = len(df)
    pos_pct = counts["Positive"] / total * 100
    neg_pct = counts["Negative"] / total * 100
    summary = (
        f"### Summary\n"
        f"- **{total}** reviews analyzed (max {MAX_ROWS})\n"
        f"- **{pos_pct:.0f}%** positive, **{neg_pct:.0f}%** negative\n"
        f"- Average predicted rating: **{df['stars'].mean():.2f} / 5**\n"
    )
    complaints = top_complaint_words(df.loc[df["sentiment"] == "Negative", text_col])
    if complaints:
        summary += "- Common words in negative reviews: " + ", ".join(f"{w} ({c})" for w, c in complaints) + "\n"

    worst = (
        df[df["sentiment"] == "Negative"]
        .sort_values(["stars", "confidence"], ascending=[True, False])
        .head(10)[[text_col, "stars", "confidence"]]
    )

    prod_col = find_product_column(df)
    if prod_col:
        products = (
            df.groupby(prod_col)
            .agg(reviews=("stars", "size"), avg_stars=("stars", "mean"),
                 pct_negative=("sentiment", lambda s: round((s == "Negative").mean() * 100)))
            .round(2)
            .sort_values("avg_stars")
            .reset_index()
        )
    else:
        products = pd.DataFrame({"note": ["Add a `product` column to see a per-product ranking."]})

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="sentiment_results_")
    df.to_csv(tmp.name, index=False)
    return summary, chart, worst, products, tmp.name


EXAMPLES = [
    ["The headphones sound amazing and the battery lasts all week. Totally worth the price!"],
    ["Stopped working after two days. Customer support never replied. Very disappointed."],
    ["It's okay. Does the job but the packaging was damaged and delivery was slow."],
    ["Bahut accha product hai, quality ekdum mast hai"],
]

with gr.Blocks(title="ReviewLens - Review Sentiment Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🛍️ ReviewLens\n"
        "Sellers get thousands of reviews. Paste one, or upload a CSV, and see what customers like and dislike."
    )
    with gr.Tabs():
        with gr.Tab("Single review"):
            inp = gr.Textbox(label="Customer review", lines=4, placeholder="Paste a product review...")
            btn = gr.Button("Analyze", variant="primary")
            verdict = gr.Markdown()
            label = gr.Label(label="Sentiment", num_top_classes=3)
            btn.click(analyze_one, inp, [label, verdict])
            gr.Examples(EXAMPLES, inp)
        with gr.Tab("Bulk CSV dashboard"):
            f = gr.File(label="Upload CSV (columns: review, optional product)", file_types=[".csv"])
            run = gr.Button("Analyze file", variant="primary")
            summary = gr.Markdown()
            plot = gr.BarPlot(x="sentiment", y="count", title="Sentiment breakdown", height=300)
            gr.Markdown("**Most negative reviews**")
            worst = gr.Dataframe(wrap=True)
            gr.Markdown("**Products ranked by average rating (lowest first)**")
            prods = gr.Dataframe()
            out_file = gr.File(label="Download results")
            run.click(analyze_csv, f, [summary, plot, worst, prods, out_file])

if __name__ == "__main__":
    demo.launch()
