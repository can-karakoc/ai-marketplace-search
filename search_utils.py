from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from huggingface_hub import InferenceClient
import os
import pandas as pd
import regex as re

HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_query(query: str) -> np.ndarray:
    return embedder.encode([query])[0]

key_amenities = [
    "hot tub", 
    "wifi", 
    "kitchen", 
    "wine glasses", 
    "air conditioning", 
    "central heating", 
    "microwave",
    "paid street parking off premises", 
    "washer", 
    "dryer", 
    "pets allowed", 
    "smoke alarm", 
    "private entrance", 
    "bathtub", 
    "first aid kit", 
    "accessible",
    'private patio or balcony'
]

AMENITY_SYNONYMS = {
    "hot tub": ["jacuzzi", "spa", "hot-tub"],
    "wifi": ["internet", "wireless"],
    "air conditioning": ["ac", "aircon"],
    "private patio or balcony": ["balcony", "terrace"],
    "pets allowed": ["pet friendly", "dog friendly"],
    "accessible": ["wheelchair", "step-free"]
}

HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

SYSTEM_PROMPT = """
You are an AI assistant that extracts structured search preferences from a user query about rental properties.

Output your answer as lines of key-value pairs ONLY, using the following keys:

location, max_price, amenities

- If a field is missing, leave it empty
- Amenities should be comma-separated
- Do NOT include extra text or explanations
"""

def extract_intent(user_input: str) -> dict:
    """
    Extract location, max_price, and amenities from a query string.
    Fully local, rule-based, no cloud LLM needed.
    """
    intent = {
        "location": None,
        "max_price": None,
        "amenities": []
    }

    # Location
    loc_match = re.search(r"\b(?:in|at)\s+([A-Za-z\s]+?)(?:\s|$|,)", user_input, re.IGNORECASE)
    if loc_match:
        intent["location"] = loc_match.group(1).strip()

    # Max price 
    price_match = re.search(r"(?:under|below|max)\s*[£$]?(\d+(?:,\d{3})*(?:\.\d+)?)", user_input, re.IGNORECASE)
    if price_match:
        intent["max_price"] = float(price_match.group(1).replace(",", ""))

    # Amenities
    found_amenities = set()
    for a in key_amenities:
        # Check for canonical name
        if a.lower() in user_input.lower():
            found_amenities.add(a)
        # Check synonyms
        for syn in AMENITY_SYNONYMS.get(a, []):
            if syn.lower() in user_input.lower():
                found_amenities.add(a)
    intent["amenities"] = list(found_amenities)

    return intent

def normalize_amenity(user_text):
    user_text = user_text.lower()
    for canonical, variants in AMENITY_SYNONYMS.items():
        if canonical in user_text:
            return canonical
        for v in variants:
            if v in user_text:
                return canonical
    return None

def final_score(row, query_embedding, requested_amenities, max_price=None):
    # Semantic similarity
    semantic_sim = cosine_similarity(
        row["description_embedding"].reshape(1, -1),
        query_embedding.reshape(1, -1)
    )[0][0]

    # Amenity matching
    if requested_amenities:
        amenity_matches = sum([row.get(a, 0) for a in requested_amenities])
        amenity_sim = amenity_matches / len(requested_amenities)
    else:
        amenity_sim = 0

    # Price penalty
    price_penalty = 0
    if max_price is not None and row["price"] > max_price:
        price_penalty = (row["price"] - max_price) / max_price

    # Combine scores
    score = semantic_sim + amenity_sim - price_penalty
    return score

def search_listings(df, user_query, key_amenities):
    # Extract intent locally
    intent = extract_intent(user_query)
    print("Intent extracted:", intent)

    requested_amenities = [
        a for a in intent.get("amenities", []) if a in key_amenities
    ]
    max_price = intent.get("max_price")
    location = intent.get("location", None)

    if location:
        location = location.strip().strip('"').strip("'")

    df = df.copy()

    # Filter by location
    if location:
        print("location filtered")
        df = df[df["city"].str.lower() == location.lower()]

    # Filter by price
    if max_price is not None:
        print("price filtered")
        df = df[df["price"] <= max_price]

    # Compute query embedding
    query_embedding = embed_query(user_query)

    # Compute final score
    df['score'] = df.apply(
        lambda row: final_score(row, query_embedding, requested_amenities, max_price),
        axis=1
    )

    # Return top 10 results
    return df.sort_values("score", ascending=False).head(10)

    