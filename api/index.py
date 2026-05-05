from functools import lru_cache
import json
import math
from pathlib import Path
import re

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

DATA_PATH = Path(__file__).with_name("listings_preview.json")
SUPPORTED_CITIES = {"london", "barcelona", "rome", "amsterdam"}
KEY_AMENITIES = [
    "hot tub",
    "wifi",
    "kitchen",
    "air conditioning",
    "washer",
    "dryer",
    "pets allowed",
    "private entrance",
    "bathtub",
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

app = FastAPI(title="AI Marketplace Search API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_listings() -> list[dict]:
    return json.loads(DATA_PATH.read_text())


def extract_intent(query: str) -> dict:
    lowered = query.lower()
    location = next(
        (city.title() for city in SUPPORTED_CITIES if re.search(rf"\b{city}\b", lowered)),
        None,
    )
    price_match = re.search(
        r"(?:under|below|max|up to|less than)\s*[£$€]?(\d+(?:,\d{3})*(?:\.\d+)?)",
        query,
        re.IGNORECASE,
    )
    max_price = float(price_match.group(1).replace(",", "")) if price_match else None

    amenities = set()
    for amenity in KEY_AMENITIES:
        if amenity in lowered:
            amenities.add(amenity)
        for synonym in AMENITY_SYNONYMS.get(amenity, []):
            if re.search(rf"\b{re.escape(synonym)}\b", lowered):
                amenities.add(amenity)

    return {"location": location, "max_price": max_price, "amenities": sorted(amenities)}


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def coerce_price(value):
    if isinstance(value, (int, float)) and not math.isnan(value):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else None


def row_score(row: dict, query_tokens: set[str], requested_amenities: list[str], max_price):
    text = " ".join(str(row.get(column, "")) for column in ["name", "description", "city"])
    row_tokens = tokenize(text)
    lexical_score = len(query_tokens & row_tokens) / max(len(query_tokens), 1)

    amenity_score = 0
    if requested_amenities:
        matches = sum(1 for amenity in requested_amenities if row.get(amenity, 0) == 1)
        amenity_score = matches / len(requested_amenities)

    price = coerce_price(row.get("price"))
    price_score = 0.15 if max_price is not None and price is not None and price <= max_price else 0
    return round(lexical_score + amenity_score + price_score, 4)


def public_listing(row: dict, score: float):
    return {
        "city": str(row.get("city", "")).title(),
        "name": str(row.get("name", "Untitled listing")),
        "price": coerce_price(row.get("price")),
        "score": score,
        "description": str(row.get("description", ""))[:320],
        "picture_url": str(row.get("picture_url", "")),
        "url": str(row.get("url", "")),
        "latitude": row.get("latitude"),
        "longitude": row.get("longitude"),
    }


@app.get("/api/health")
def health():
    return {"ok": True, "service": "ai-marketplace-search", "records": len(load_listings())}


@app.get("/api/search")
def search(q: str = Query(..., min_length=2), limit: int = Query(8, ge=1, le=20)):
    intent = extract_intent(q)
    listings = load_listings()

    if intent["location"]:
        listings = [row for row in listings if row.get("city", "").lower() == intent["location"].lower()]

    if intent["max_price"] is not None:
        listings = [
            row for row in listings
            if (coerce_price(row.get("price")) or float("inf")) <= intent["max_price"]
        ]

    query_tokens = tokenize(q)
    scored = [
        (row_score(row, query_tokens, intent["amenities"], intent["max_price"]), row)
        for row in listings
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    results = [public_listing(row, score) for score, row in scored[:limit]]
    return {"query": q, "intent": intent, "count": len(results), "results": results}
