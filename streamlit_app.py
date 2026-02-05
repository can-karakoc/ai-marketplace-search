import streamlit as st
import pandas as pd
import pydeck as pdk
from search_utils import (
    search_listings,
    embed_query,
    key_amenities,
    normalize_amenity,
    extract_intent
)

# ------------------------------------------------------------
# Page Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Marketplace Search",
    layout="wide"
)

st.title("🏠 Airbnb Marketplace Search")
st.markdown("Cities available: London, Barcelona, Rome, Amsterdam")
st.markdown("Search rental listings using **LLM-free local intent extraction + embeddings**.")

# ------------------------------------------------------------
# Load Data (Cached)
# ------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_pickle("data/processed/all_listings.pkl")

all_listings = load_data()

# ------------------------------------------------------------
# Safe Intent Wrapper (Local)
# ------------------------------------------------------------
def safe_extract_intent(query):
    try:
        return extract_intent(query)
    except Exception as e:
        st.warning("⚠️ Local intent extraction failed, running semantic-only search.")
        return {
            "location": None,
            "max_price": None,
            "amenities": []
        }

# ------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------
st.sidebar.header("⚙️ Search Controls")
max_results = st.sidebar.slider(
    "Max results to display", min_value=5, max_value=20, value=10
)
show_intent = st.sidebar.checkbox("Show extracted intent", value=True)

# ------------------------------------------------------------
# User Input
# ------------------------------------------------------------
user_query = st.text_input(
    "Describe what you're looking for:",
    placeholder="3-bedroom cottage with a hot tub and garden"
)

# ------------------------------------------------------------
# Search Action
# ------------------------------------------------------------
if st.button("🔍 Search") and user_query.strip():

    with st.spinner("Searching listings..."):

        # --- Extract intent locally
        intent = safe_extract_intent(user_query)
        location = intent.get("location")
        max_price = intent.get("max_price")
        amenities = intent.get("amenities", [])

        # --- Show intent (for recruiters / WOW factor)
        if show_intent:
            st.info(
                f"""
                **🧠 Intent Extracted**
                - Location: {location or "Any"}
                - Max Price: {max_price or "None"}
                - Amenities: {", ".join(amenities) if amenities else "None"}
                """
            )

        # --- Run search engine
        top_results = search_listings(all_listings, user_query, key_amenities)
        top_results = top_results.head(max_results).copy()

        if len(top_results) == 0:
            st.warning("No results found.")
            st.stop()

        # ------------------------------------------------------------
        # Add Ranking Column
        # ------------------------------------------------------------
        top_results["rank"] = range(1, len(top_results) + 1)

        # ------------------------------------------------------------
        # MAP VIEW (Ranked Pins with Hover)
        # ------------------------------------------------------------
        st.subheader("📍 Map View (Ranked Results)")

        map_data = top_results[
            ["latitude", "longitude", "name", "price", "score", "rank"]
        ].dropna()

        # Scatter layer: bigger points with auto highlight
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position="[longitude, latitude]",
            get_radius=180,  # bigger points
            get_fill_color=[255, 140, 0, 200],
            pickable=True,
            auto_highlight=True,
        )

        # Text layer: rank numbers
        text_layer = pdk.Layer(
            "TextLayer",
            data=map_data,
            get_position="[longitude, latitude]",
            get_text="rank",
            get_size=16,
            get_color=[0, 0, 0],
            get_alignment_baseline="'center'",
        )

        # Tooltip HTML
        tooltip = {
            "html": """
            <b>#{rank}: {name}</b><br/>
            💷 £{price}<br/>
            ⭐ Score: {score}
            """,
            "style": {"backgroundColor": "white", "color": "black"}
        }

        # Render map
        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=pdk.ViewState(
                    latitude=map_data["latitude"].mean(),
                    longitude=map_data["longitude"].mean(),
                    zoom=11,
                    pitch=0
                ),
                layers=[scatter_layer, text_layer],
                tooltip=tooltip
            )
        )

        # ------------------------------------------------------------
        # TABLE VIEW
        # ------------------------------------------------------------
        st.subheader("📊 Top Ranked Matches")
        display_df = top_results[["rank", "city", "name", "price", "score"]].copy()
        st.dataframe(display_df, use_container_width=True)

        # ------------------------------------------------------------
        # LISTING CARDS WITH IMAGES
        # ------------------------------------------------------------
        st.subheader("🏡 Listing Details")
        for _, row in top_results.iterrows():

            with st.container():
                st.markdown(
                    f"## #{row['rank']} — {row['name']}  (£{row['price']})"
                )

                col1, col2 = st.columns([1, 2])

                # Image
                with col1:
                    if "picture_url" in row and row["picture_url"]:
                        st.image(row["picture_url"], use_container_width=True)
                    else:
                        st.write("No image available")

                # Info
                with col2:
                    st.write(f"📍 **City:** {row['city']}")
                    st.write(f"⭐ **Score:** {row['score']:.3f}")

                    if "amenities" in row:
                        st.write(
                            "✅ **Amenities:** " + ", ".join(row["amenities"][:8])
                        )

                    if "description" in row:
                        st.write(
                            "📝 **Description:** " + row["description"][:250] + "..."
                        )

                    if "url" in row and row["url"]:
                        st.markdown(
                            f"🔗 [View Original Listing]({row['url']})"
                        )

                st.markdown("---")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.markdown(
    "Built using Streamlit + Semantic Search + Intent Extraction"
)