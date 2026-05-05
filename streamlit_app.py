import ast
import io
import os
import urllib.request

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pydeck as pdk
import streamlit as st

from search_utils import extract_intent, key_amenities, search_listings

# ------------------------------------------------------------
# Page Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Marketplace Search",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Airbnb Marketplace Search")
st.markdown("Cities available: London, Barcelona, Rome, Amsterdam")
st.markdown("Search rental listings using **local intent extraction + semantic embeddings**.")

DATA_URL = "https://huggingface.co/datasets/cankarakoc/ai-marketplace-search/resolve/main/listings.parquet"
DEFAULT_SAMPLE_ROWS = int(os.environ.get("APP_SAMPLE_ROWS", "500"))


def get_secret(name: str):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


MAPBOX_API_KEY = os.environ.get("MAPBOX_API_KEY") or get_secret("MAPBOX_API_KEY")


@st.cache_data(show_spinner="Downloading listings...")
def load_data(n_rows: int = DEFAULT_SAMPLE_ROWS) -> pd.DataFrame:
    """Load a small product demo slice from the hosted listings dataset."""
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        file_bytes = io.BytesIO(response.read())

    table = pq.read_table(file_bytes)
    df = table.to_pandas().head(n_rows)

    def str_to_array(value):
        if isinstance(value, str):
            value = value.replace("[", "").replace("]", "").strip()
            value = ",".join(value.split())
            return np.fromstring(value, sep=",", dtype=np.float32)
        return np.array(value, dtype=np.float32)

    df["description_embedding"] = df["description_embedding"].apply(str_to_array)
    return df


@st.cache_data
def normalize_amenities(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else [value]
        except Exception:
            return [value]
    return []


def safe_extract_intent(query: str) -> dict:
    try:
        return extract_intent(query)
    except Exception:
        st.warning("Local intent extraction failed, running semantic-only search.")
        return {"location": None, "max_price": None, "amenities": []}


def render_intent(intent: dict) -> None:
    st.info(
        f"""
        **🧠 Intent Extracted**
        - Location: {intent.get("location") or "Any"}
        - Max Price: {intent.get("max_price") or "None"}
        - Amenities: {", ".join(intent.get("amenities", [])) if intent.get("amenities") else "None"}
        """
    )


def render_map(top_results: pd.DataFrame) -> None:
    st.subheader("📍 Map View")

    if not MAPBOX_API_KEY:
        st.caption("Set MAPBOX_API_KEY to enable the interactive map in production.")
        return

    map_data = top_results[["latitude", "longitude", "name", "price", "score", "rank"]].dropna()
    if map_data.empty:
        st.caption("No mappable listings in these results.")
        return

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position=["longitude", "latitude"],
        get_radius=180,
        get_fill_color=[255, 140, 0, 200],
        pickable=True,
        auto_highlight=True,
    )

    text_layer = pdk.Layer(
        "TextLayer",
        data=map_data,
        get_position=["longitude", "latitude"],
        get_text="rank",
        get_size=16,
        get_color=[0, 0, 0],
        get_alignment_baseline="center",
    )

    tooltip = {
        "html": "<b>#{rank}: {name}</b><br/>💷 £{price}<br/>⭐ Score: {score}",
        "style": {"backgroundColor": "white", "color": "black"},
    }

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=map_data["latitude"].mean(),
                longitude=map_data["longitude"].mean(),
                zoom=11,
                pitch=0,
            ),
            layers=[scatter_layer, text_layer],
            tooltip=tooltip,
            api_keys={"mapbox": MAPBOX_API_KEY},
        )
    )


# ------------------------------------------------------------
# Data + Sidebar Controls
# ------------------------------------------------------------
all_listings = load_data()
st.caption(f"Loaded {len(all_listings):,} demo listings")

st.sidebar.header("⚙️ Search Controls")
max_results = st.sidebar.slider("Max results to display", min_value=5, max_value=20, value=10)
show_intent = st.sidebar.checkbox("Show extracted intent", value=True)

# ------------------------------------------------------------
# User Input
# ------------------------------------------------------------
user_query = st.text_input(
    "Describe what you're looking for:",
    placeholder="3-bedroom cottage in London under £200 with a hot tub and garden",
)

# ------------------------------------------------------------
# Search Action
# ------------------------------------------------------------
if st.button("🔍 Search") and user_query.strip():
    with st.spinner("Searching listings..."):
        intent = safe_extract_intent(user_query)

        if show_intent:
            render_intent(intent)

        top_results = search_listings(all_listings, user_query, key_amenities, intent=intent)
        top_results = top_results.head(max_results).copy()

        if top_results.empty:
            st.warning("No results found. Try broadening the city, price, or amenity request.")
            st.stop()

        top_results["rank"] = range(1, len(top_results) + 1)

        render_map(top_results)

        st.subheader("📊 Top Ranked Matches")
        display_df = top_results[["rank", "city", "name", "price", "score"]].copy()
        display_df["score"] = display_df["score"].round(3)
        st.dataframe(display_df, use_container_width=True)

        st.subheader("🏡 Listing Details")
        for _, row in top_results.iterrows():
            with st.container():
                st.markdown(f"## #{row['rank']} — {row['name']} (£{row['price']})")

                col1, col2 = st.columns([1, 2])

                with col1:
                    picture_url = row.get("picture_url")
                    if isinstance(picture_url, str) and picture_url.strip():
                        st.image(picture_url, use_container_width=True)
                    else:
                        st.write("No image available")

                with col2:
                    st.write(f"📍 **City:** {row['city']}")
                    st.write(f"⭐ **Score:** {row['score']:.3f}")

                    amenities = normalize_amenities(row.get("amenities"))
                    if amenities:
                        st.write("✅ **Amenities:** " + ", ".join(map(str, amenities[:8])))

                    description = row.get("description")
                    if isinstance(description, str) and description.strip():
                        st.write("📝 **Description:** " + description[:250] + "...")

                    listing_url = row.get("url")
                    if isinstance(listing_url, str) and listing_url.strip():
                        st.markdown(f"🔗 [View Original Listing]({listing_url})")

                st.markdown("---")

st.markdown("---")
st.markdown("Built using Streamlit + semantic search + local intent extraction")
