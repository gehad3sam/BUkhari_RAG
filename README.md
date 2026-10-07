# BUkhari_RAG

# Sahih al-Bukhari AI Search Engine 🕌

[](https://www.google.com/search?q=https://elbukhari.streamlit.app/) *(Replace this link with your actual Streamlit URL)*

An intelligent, Arabic-language search engine built to retrieve and explain Hadiths from Sahih al-Bukhari. This application uses an advanced Retrieval-Augmented Generation (RAG) pipeline to understand semantic queries, find the most relevant Hadiths, and synthesize accurate, context-aware answers.

## 🧠 Architecture & Request Flow

When a user asks a question, the system processes it through the following pipeline:

1. **Query Embedding:** The user's Arabic query is converted into a high-dimensional vector using **Cohere's Multilingual Embedding Model**.
2. **Vector Retrieval:** The vector is queried against a **Qdrant Vector Database** containing the pre-embedded Sahih al-Bukhari dataset, retrieving the top broad matches.
3. **Reranking:** **Cohere's Reranker** evaluates and re-orders the retrieved Hadiths based on exact semantic relevance to the user's prompt.
4. **Generation:** The optimized context is passed to an open-source LLM via **Groq's LPU Inference Engine**, generating a sub-second, highly accurate Arabic response.

## 🛠️ Tech Stack & Rationale

* **Frontend (Streamlit):** Chosen for rapid, lightweight deployment of Python-based AI applications.
* **Vector Database (Qdrant):** Cloud-hosted vector search that effortlessly handles thousands of dense vectors with low-latency retrieval.
* **Embeddings & Reranking (Cohere):** `embed-multilingual-v3.0` offers state-of-the-art semantic understanding for the Arabic language, significantly outperforming standard English-centric models.
* **LLM Inference (Groq):** Provides near-instantaneous token generation, ensuring a real-time, conversational user experience.

## 🚀 Local Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/BUkhari_RAG.git
cd BUkhari_RAG

```

**2. Create a virtual environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt

```

**3. Configure Environment Variables**
Create a `.env` file in the root directory and add your API keys:

```env
QDRANT_URL="your-qdrant-cluster-url"
QDRANT_API_KEY="your-qdrant-api-key"
GROQ_API_KEY="your-groq-api-key"
COHERE_API_KEY="your-cohere-api-key"

```

**4. Run the Application**

```bash
streamlit run app.py

```

## ⚠️ Disclaimer

This application uses artificial intelligence to retrieve and summarize information. While efforts are made to ensure accuracy via the RAG architecture, LLMs can occasionally hallucinate. The original Arabic text of Sahih al-Bukhari remains the sole authoritative source.

---

### How to use this:

1. Copy the Markdown code above.
2. Open the `README.md` file in your VS Code.
3. Paste the code, replacing the placeholder GitHub and Streamlit URLs with your actual links.
4. Run your Git commands (`git add README.md`, `git commit -m "Update README for portfolio"`, `git push origin main`).

With the UI polished and the repository looking professional, the final step is preparing your LinkedIn post and demo video. Let me know when you are ready to craft the launch post!