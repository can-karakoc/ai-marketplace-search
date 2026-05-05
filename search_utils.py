from functools import lru_cache
import logging
import os

import numpy as np
import regex as re
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed_query(query: str) -> np.ndarray:
    return get_embedder().encode([query])[0]


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
    "private patio or balcony",
]

AMENITY_SYNONYMS = {
    "hot tub": ["jacuzzi", "spa", "hot-tub"],
    "wifi": ["internet", "wireless"],
    "air conditioning": ["ac", "aircon"],
    "private patio or balcony": ["balcony", "terrace", "garden"],
    "pets allowed": ["pet friendly", "dog friendly", "pets", "dogs"],
    "accessible": ["wheelchair", "step-free"],
}

SUPPORTED_CITIES = {"london", "barcelona", "rome", "amsterdam"}


def extract_intent(user_input: str) -> dict:
    """Extract location, max_price, and amenities with lightweight local rules."""
    intent = {"location": None, "max_price": None, "amenities": []}
    query = user_input.lower()

    for city in SUPPORTED_CITIES:
        if re.search(rf"\b{re.escape(city)}\b", query):
            intent["location"] = city.title()
            break

    price_match = re.search(
        r"(?:under|below|max|up to|less than)\s*[£$€]?(\d+(?:,\d{3})*(?:\.\d+)?)",
        user_input,
        re.IGNORECASE,
    )
    if price_match:
        intent["max_price"] = float(price_match.group(1).replace(",", ""))

    found_amenities = set()
    for amenity in key_amenities:
        if amenity.lower() in query:
            found_amenities.add(amenity)
        for synonym in AMENITY_SYNONYMS.get(amenity, []):
            if re.search(rf"\b{re.escape(synonym.lower())}\b", query):
                found_amenities.add(amenity)

    intent["amenities"] = sorted(found_amenities)
    return intent


def normalize_amenity(user_text: str):
    user_text = user_text.lower()
    for canonical, variants in AMENITY_SYNONYMS.items():
        if canonical in user_text:
            return canonical
        for variant in variants:
            if variant in user_text:
                return canonical
    return None


def final_score(row, query_embedding, requested_amenities, max_price=None):
    semantic_sim = cosine_similarity(
        row["description_embedding"].reshape(1, -1),
        query_embedding.reshape(1, -1),
    )[0][0]

    if requested_amenities:
        amenity_matches = sum(row.get(amenity, 0) for amenity in requested_amenities)
        amenity_sim = amenity_matches / len(requested_amenities)
    else:
        amenity_sim = 0

    price_penalty = 0
    if max_price is not None and row["price"] > max_price:
        price_penalty = (row["price"] - max_price) / max_price

    return semantic_sim + amenity_sim - price_penalty


def search_listings(df, user_query, amenities, intent=None):
    intent = intent or extract_intent(user_query)
    logger.info("Intent extracted: %s", intent)

    requested_amenities = [
        amenity for amenity in intent.get("amenities", []) if amenity in amenities
    ]
    max_price = intent.get("max_price")
    location = intent.get("location")

    results = df.copy()

    if location:
        results = results[results["city"].str.lower() == location.lower()]

    if max_price is not None:
        results = results[results["price"] <= max_price]

    if results.empty:
        return results.assign(score=[])

    query_embedding = embed_query(user_query)
    results["score"] = results.apply(
        lambda row: final_score(row, query_embedding, requested_amenities, max_price),
        axis=1,
    )

    return results.sort_values("score", ascending=False)
