# ReviewLens - E-commerce Review Sentiment Analyzer

Sellers receive thousands of reviews and cannot tell which products customers like or dislike.
ReviewLens wraps a pre-trained Hugging Face model in a simple web app that anyone can try.

**Live demo:** (link coming soon)

## Model
`nlptown/bert-base-multilingual-uncased-sentiment` from Hugging Face. It predicts a 1-5 star rating, which the app turns into Positive, Neutral or Negative.

## Features
- Paste one review and get its sentiment and star rating
- Upload a CSV (`review` column, optional `product` column) to see a dashboard:
  sentiment breakdown, common words in negative reviews, most negative reviews, and products ranked by rating
- Download the labeled results as a CSV

## Tech stack
Python, Hugging Face Transformers, Gradio, Pandas

## Run locally
```bash
pip install -r requirements.txt gradio
python app.py
