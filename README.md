# 🏘️ AI Marketplace Search

Conversational search for marketplace-style rental listings. The app combines local intent extraction, sentence embeddings, amenity matching, and price-aware ranking to turn a natural-language request into ranked Airbnb-style results.

## Features

- 🔍 Semantic search with `all-MiniLM-L6-v2` embeddings
- 🧠 Local intent extraction for city, price ceiling, and amenities
- 🏷️ Amenity synonym matching and scoring
- 💸 Price-aware ranking
- 🗺️ Optional Mapbox-powered map view
- 🖥️ Streamlit product demo UI

## Project Structure

```text
.
├── search_utils.py        # Core search, intent extraction, and scoring logic
├── streamlit_app.py       # Streamlit UI
├── notebook.ipynb         # Exploration and experiments
├── requirements.txt       # Runtime dependencies
└── README.md
```

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Optional environment variables:

```bash
export MAPBOX_API_KEY="your_mapbox_token"   # Enables the interactive map
export APP_SAMPLE_ROWS=500                   # Controls hosted dataset sample size
export EMBEDDING_MODEL="all-MiniLM-L6-v2"    # Overrides the local embedding model
```

The app downloads a parquet demo dataset from Hugging Face at runtime, so no local data checkout is required.

## Deployment Notes

The current application is a Streamlit app. Streamlit expects a long-running web process, while Vercel's Python support is designed around serverless `/api` handlers and ASGI apps such as FastAPI. For a production Vercel launch, the recommended path is to split the product into:

1. A Vercel-hosted web frontend, ideally Next.js.
2. A Vercel Python/FastAPI search API or a separate hosted search service.
3. A persisted vector index or precomputed embeddings store so cold starts stay fast.

For the fastest live demo without replatforming, deploy this repository to Streamlit Community Cloud, Hugging Face Spaces, Render, or another host that supports persistent Streamlit processes.
