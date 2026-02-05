# 🏘️ AI Marketplace Search

---

## Features

- 🔍 **Semantic Search** using sentence embeddings (MiniLM)
- 🧠 **LLM Intent Extraction** (location, price, amenities)
- 🏷️ **Amenity Matching** with normalization & scoring
- 💸 **Price-Aware Ranking**
- 🖥️ **Interactive Streamlit UI**
- 🧩 Modular, extensible architecture

---
## 📂 Project Structure

```text
.
├── search_utils.py        # Core logic: embeddings, intent extraction, scoring
├── streamlit_app.py       # Streamlit UI
├── notebook.ipynb         # Exploration, debugging, and experiments
├── data/
│   └── processed
│   └── raw
├── requirements.txt
└── README.md
```

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
```

#### For macOS / Linux
```bash
source .venv/bin/activate
```

#### For Windows
```bash
.venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a Hugging Face API Token:
1. Go to [Hugging Face tokens](https://huggingface.co/settings/tokens)
2. Generate a token with read access

```bash
streamlit run streamlit_app.py
```

#### For macOS / Linux
```bash
export HF_TOKEN="your_hf_token_here" 
```

#### For Windows
```bash
set HF_TOKEN=your_hf_token_here 
```
---