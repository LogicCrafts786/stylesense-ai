# stylesense-ai
# 👗 StyleSense AI – Multi-Modal Conversational Shopping Agent

[

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

](https://www.python.org/)
[

![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red.svg)

](https://streamlit.io/)
[

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

](LICENSE)
[

![Gemini](https://img.shields.io/badge/Google-Gemini%20API-4285F4.svg)

](https://ai.google.dev/)

> An AI-powered personal shopping assistant that understands natural language, analyzes outfit photos, and recommends complete, budget-aware, occasion-appropriate outfits — with reasoning you can actually follow.

---

## 🧵 Table of Contents

- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [Architecture Overview](#-architecture-overview)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Configuration](#-configuration)
- [Usage Examples](#-usage-examples)
- [Testing](#-testing)
- [Screenshots](#-screenshots)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Problem Statement

Traditional e-commerce search is keyword-driven: type "blue shirt," get a wall of blue shirts. It can't answer *"What should I wear to a beach wedding in October if I have a $150 budget and want to look sharp but not overdressed?"*

**StyleSense AI** solves this by combining:
- LLM-based intent understanding (Gemini)
- Multi-modal reasoning (text + image)
- Structured retrieval (vector search over product catalog)
- Domain logic (color theory, weather/occasion mapping, budget constraints)

into a single conversational agent that reasons like a personal stylist.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 💬 **Conversational Assistant** | Multi-turn chat with persistent memory of preferences and past turns |
| 🖼️ **Image Analysis** | Upload a photo of an item or outfit; Gemini Vision extracts color, style, category |
| 👕 **Outfit Recommendations** | Full outfit generation (top, bottom, shoes, accessories) with rationale |
| 🎨 **Color & Style Matching** | Rule-based + LLM color harmony analysis (complementary, analogous, etc.) |
| 💰 **Budget-Aware Filtering** | Recommendations constrained to user-specified budget ranges |
| 🌦️ **Occasion & Weather Reasoning** | Adjusts recommendations for climate and event type |
| ⚖️ **Product Comparison** | Side-by-side comparison of multiple products on price/quality/style fit |
| 📝 **Review Summarization** | Scrapes and summarizes product reviews into pros/cons |
| 🛍️ **Bundle Generation** | Assembles complete "shop the look" bundles |
| 🔁 **Alternative Recommendations** | Offers substitutes when budget or availability doesn't match |

---

## 🏗️ Architecture Overview

StyleSense AI uses a **LangGraph state machine** to orchestrate a single conversational agent that can call specialized tools. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full diagrams and design rationale.

```
User Input (text/image)
        │
        ▼
┌───────────────────┐
│  Streamlit UI      │
└─────────┬──────────┘
          │
          ▼
┌───────────────────────────┐
│  LangGraph Shopping Agent  │◄──────► Conversation Memory (Chroma/FAISS)
└─────────┬──────────────────┘
          │
   ┌──────┴───────────────────────────────┐
   ▼             ▼           ▼            ▼
Image Tool   Search Tool  Color Tool  Budget/Weather Tools
   │             │           │            │
   └──────┬──────┴─────┬─────┴────────────┘
          ▼             ▼
   Gemini Service   Product Catalog / Scraper Service
```

---

## 🛠️ Tech Stack

- **UI:** Streamlit
- **LLM/Vision:** Google Gemini API (`gemini-1.5-pro`, `gemini-1.5-pro-vision`)
- **Orchestration:** LangChain + LangGraph
- **Vector Store:** ChromaDB (default) or FAISS
- **Scraping:** BeautifulSoup + Requests
- **Image Handling:** Pillow
- **Data:** Pandas, JSON sample datasets
- **Config:** python-dotenv, pydantic-settings
- **Testing:** pytest, pytest-cov, pytest-mock

---

## 📂 Project Structure

See the full tree in the repository root. High-level layout:

```
stylesense-ai/
├── app.py              # Streamlit entrypoint
├── src/
│   ├── agents/         # LangGraph orchestration
│   ├── tools/           # LangChain tools (image, color, budget, etc.)
│   ├── memory/          # Conversation + vector memory
│   ├── prompts/         # Prompt templates
│   ├── services/         # External API wrappers (Gemini, weather, scraper)
│   ├── models/           # Data schemas
│   ├── utils/            # Logging, config, validation
│   └── ui/               # Streamlit UI components
├── data/                 # Sample product/review datasets
├── docs/                 # Architecture & setup docs
├── tests/                # Unit tests
└── screenshots/          # App screenshots
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A [Google Gemini API key](https://ai.google.dev/)
- (Optional) An OpenWeatherMap API key for live weather reasoning

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/stylesense-ai.git
cd stylesense-ai

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 5. Run the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## ⚙️ Configuration

All configuration is managed via `.env` and loaded through `src/utils/config.py` using `pydantic-settings`. See [`.env.example`](.env.example) for the full list of variables, including:

- `GEMINI_API_KEY` – **required**
- `GEMINI_MODEL_NAME` / `GEMINI_VISION_MODEL_NAME`
- `VECTOR_STORE_TYPE` – `chroma` or `faiss`
- `WEATHER_API_KEY` – optional, enables live weather reasoning
- `MAX_CONVERSATION_HISTORY`, `MAX_UPLOAD_SIZE_MB`

---

## 💡 Usage Examples

**Text-only query:**
> "I need an outfit for a rooftop cocktail party in September, budget $200, I like earthy tones."

**Image + text query:**
> Upload a photo of a jacket → "What can I pair with this for a casual weekend brunch?"

**Comparison:**
> "Compare the Chelsea boots and the derby shoes you showed me — which is better value?"

**Bundle generation:**
> "Build me a full capsule wardrobe for a 5-day business trip, budget $500."

---

## 🧪 Testing

```bash
# Run all tests with coverage
pytest

# Run a specific test module
pytest tests/test_tools/test_color_matching_tool.py -v

# Run only unit tests (skip integration tests requiring API keys)
pytest -m unit
```

---

## 🖼️ Screenshots

| Chat Interface | Image Upload | Outfit Recommendation |
|---|---|---|
| 

![Chat](screenshots/chat_interface.png)

 | 

![Upload](screenshots/image_upload_demo.png)

 | 

![Outfit](screenshots/outfit_recommendation.png)

 |

---

## 🗺️ Roadmap

- [ ] Real-time inventory integration via retailer APIs
- [ ] User authentication and persistent long-term profiles
- [ ] Multi-language support
- [ ] Mobile-optimized UI
- [ ] A/B testing framework for recommendation quality

---

## 🤝 Contributing

Contributions are welcome! Please see [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) for guidelines on code style, branching, and PR process.

---

## 📜 License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.
