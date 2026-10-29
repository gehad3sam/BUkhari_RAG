# 🕌 الباحث الذكي في مختصر صحيح البخاري | AskBukhari RAG

محرك بحث ومساعد ذكي يعتمد على معمارية **الاسترجاع المعزز بالتوليد (Two-Stage RAG)** للإجابة عن الأسئلة الشرعية والحديثية والتاريخية بالاعتماد **حصرياً** على كتاب **«التجريد الصريح لأحاديث الجامع الصحيح»** (مختصر صحيح البخاري للإمام الزبيدي — أكثر من 2,200 حديث)، مع توثيق مزدوج ومنع صارم للهلوسة.

* 🌐 **تجربة التطبيق مباشرة (Live App):** [https://askbukhari.streamlit.app/](https://askbukhari.streamlit.app/)
* 💻 **المستودع البرمجي (GitHub Repo):** [https://github.com/gehad3sam/BUkhari_RAG](https://github.com/gehad3sam/BUkhari_RAG)

---

## 📑 فهرس المحتويات (Table of Contents)

1. [⚙️ كيف يعمل النظام (معمارية الكود التفاعلية)](#1-كيف-يعمل-النظام-معمارية-الكود-التفاعلية)
2. [🚀 كيفية تشغيل التطبيق محلياً خطوة بخطوة (Installation & Run Guide)](#2-كيفية-تشغيل-التطبيق-محليا-خطوة-بخطوة)
3. [☁️ كيفية النشر على Streamlit Cloud](#3-كيفية-النشر-على-streamlit-cloud)
4. [💡 أبرز التحديات الهندسية وكيف تم حلها](#4-أبرز-التحديات-الهندسية-وكيف-تم-حلها)
5. [🇬🇧 English Documentation & Setup Guide](#5-english-documentation--setup-guide)

---

## 1. كيف يعمل النظام (معمارية الكود التفاعلية)

يمر سؤال المستخدم بثلاث مراحل برمجية متتالية لضمان الدقة والسرعة (أقل من 3 ثوانٍ):

```python
# المرحلة 1: البحث الدلالي في Qdrant عبر تضمينات Cohere
query_vector = embedding_model.embed_query(user_query)
search_results = qdrant_client.query_points(
    collection_name="sahih_bukhari_with_cohere",
    query=query_vector,
    limit=15
)

# فلتر الأمان المبكر: رفض الأسئلة الخارجة عن النطاق لتوفير استهلاك API وتقليل Latency
SIMILARITY_THRESHOLD = 0.42
candidate_chunks = [p for p in search_results.points if p.score >= SIMILARITY_THRESHOLD]

# المرحلة 2: إزالة التكرار (Deduplication) وإعادة الترتيب الدقيق عبر Cohere Reranker v3
docs = [
    f"كتاب: {p.payload['book_title']} | {p.payload['original_text']}"
    for p in unique_candidates
]
reranked = cohere_client.rerank(
    model="rerank-multilingual-v3.0",
    query=user_query,
    documents=docs,
    top_n=min(7, len(docs))
)

# المرحلة 3: التوليد بالبث الحي (Streaming) عبر Gemini 3.5 Flash-Lite
response_stream = llm_client.models.generate_content_stream(
    model="gemini-3.5-flash-lite",
    contents=full_prompt,
    config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=2048,
        system_instruction=system_prompt,
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.MINIMAL
        )
    )
)
```

---

## 2. كيفية تشغيل التطبيق محلياً خطوة بخطوة

لتشغيل التطبيق على جهازك الشخصي، اتبع الخطوات التالية بالترتيب:

### الخطوة 1: استنساخ المستودع (Clone Repository)

افتح الطرفية (`Terminal` أو `PowerShell`) ونفّذ الأمر التالي:

```bash
git clone https://github.com/gehad3sam/BUkhari_RAG.git
cd BUkhari_RAG
```

### الخطوة 2: إنشاء وتفعيل البيئة الافتراضية (Virtual Environment)

**على نظام Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**على نظام Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### الخطوة 3: تثبيت المكتبات المطلوبة (`requirements.txt`)

```bash
pip install -r requirements.txt
```

محتويات ملف `requirements.txt` المعتمدة في المشروع:

```txt
streamlit>=1.35.0
python-dotenv>=1.0.0
qdrant-client>=1.10.0
cohere>=5.5.0
langchain-cohere>=0.1.5
google-genai>=1.0.0
```

### الخطوة 4: إعداد مفاتيح الـ API (ملف `.env`)

أنشئ ملفاً باسم `.env` في المجلد الرئيسي للمشروع، وأضف بداخله المفاتيح الأربعة التالية:

```ini
QDRANT_URL="https://your-cluster-url.qdrant.io"
QDRANT_API_KEY="your_qdrant_api_key_here"
COHERE_API_KEY="your_cohere_api_key_here"
GEMINI_API_KEY="your_gemini_api_key_here"
```

* **روابط الحصول على المفاتيح:**
  * مفتاح **Qdrant Cloud**: من [cloud.qdrant.io](https://cloud.qdrant.io/)
  * مفتاح **Cohere API**: من [dashboard.cohere.com](https://dashboard.cohere.com/api-keys)
  * مفتاح **Gemini API**: من [aistudio.google.com](https://aistudio.google.com/app/apikey)

### الخطوة 5: تشغيل تطبيق Streamlit

```bash
streamlit run app.py
```

سيفتح التطبيق تلقائياً في متصفحك على العنوان: `http://localhost:8501`

---

## 3. كيفية النشر على Streamlit Cloud

عند رفع التطبيق على منصة [Streamlit Community Cloud](https://share.streamlit.io/)، لا ترفع ملف `.env` إلى GitHub، بل أضف المفاتيح داخل إعدادات التطبيق (**Settings -> Secrets**) بصيغة `TOML` التالية:

```toml
QDRANT_URL = "https://your-cluster-url.qdrant.io"
QDRANT_API_KEY = "your_qdrant_api_key_here"
COHERE_API_KEY = "your_cohere_api_key_here"
GEMINI_API_KEY = "your_gemini_api_key_here"
```

---

## 4. أبرز التحديات الهندسية وكيف تم حلها

* **تداخل الأحداث والشخصيات في الحديث الواحد:**
  مثل ورود غزوة بدر داخل حديث يتحدث عن غزوة تبوك، أو تعدد الأسماء في الأحاديث الطويلة (كحديث الإفك) بين راوٍ وشاهد ومستشار وفاعل، بالإضافة إلى الألفاظ التراثية ذات المعاني الخاصة. تم حل هذه التحديات عبر أكثر من **6 جولات من التقييم المنهجي وتحليل الأخطاء (Error Analysis)**، وبناء قواعد تحليل نحوي ودلالي صارمة داخل الـ `System Prompt` تفصل بصمت بين الأدوار والأحداث وتمنع الاستنتاج من خارج النص.
* **الفجوة بين سؤال المستخدم بالعامية ولغة الحديث:**
  تمت تجربة إضافة مرحلة **Query Expansion** عبر الـ LLM لتوليد مرادفات تراثية قبل البحث، لكن التجربة العملية أظهرت أنها تضاعف زمن الاستجابة (**Latency**) وتستهلك حدود الـ **Rate Limits** دون تحسن كبير؛ فتم الاستغناء عنها لصالح تقوية مرحلة الـ **Reranking** عبر دمج عنوان الكتاب مع متن الحديث.
* **الموازنة بين الـ Latency والدقة ومقاومة الـ Prompt Injection:**
  * **في مرحلة الـ Embeddings:** تم استبدال نموذج `BGE-M3` بسبب بطء التحميل والاستجابة بنموذج `Cohere (embed-multilingual-v3.0)` السحابي.
  * **في مرحلة الـ LLM:** بعد المقارنة العملية بين النماذج المتوفرة على منصة **Groq** (`Llama-3.3-70B` و`GPT-OSS-120B` و`Qwen` الذي كان الأسرع لكن ظهرت فيه أخطاء في الفهم السياقي للنصوص العربية)، استقر الاختيار على **`Gemini 3.5 Flash-Lite`** مع إعداد `ThinkingLevel.MINIMAL` لدقته العالية في فهم النصوص التراثية وسرعته في الرد وصموده أمام اختبارات الـ **Prompt Injection** والأسئلة المفخخة.

---

## 5. English Documentation & Setup Guide

### 📖 Overview

**AskBukhari** is a production-grade **Retrieval-Augmented Generation (RAG)** web application engineered for **Classical Arabic Hadith literature** (*Mukhtasar Sahih al-Bukhari* by Imam Al-Zubaidi, **2,200+ authenticated hadiths**).

* **Live Demo:** [https://askbukhari.streamlit.app/](https://askbukhari.streamlit.app/)
* **Tech Stack:** `Python` | `Streamlit` (Mobile-First RTL UI) | `Qdrant Vector DB` | `Cohere Embeddings & Rerank v3` | `Google Gemini 3.5 Flash-Lite`

### ⚙️ Architecture Highlights

| Stage | Component | Configuration & Purpose |
| :--- | :--- | :--- |
| **1. Dense Retrieval** | `Qdrant` + `Cohere embed-multilingual-v3.0` | Retrieves Top-15 candidates with an early-exit similarity threshold (`>= 0.42`) to block out-of-scope queries and save API tokens. |
| **2. Contextual Reranking** | `Cohere rerank-multilingual-v3.0` | Deduplicates chunks, prepends `Book Title` to `Hadith Text`, and filters top passages (`top_n = 7`, `relevance_score >= 0.10`). |
| **3. Streaming Generation** | `Gemini 3.5 Flash-Lite` | Streams responses at `temperature = 0.0` with `ThinkingLevel.MINIMAL` for low latency (~2–3s) and enforces dual citation (Mukhtasar + original Bukhari numbers). |

### 🛠️ Quick Start (English)

**1. Clone the repo & create a virtual environment:**

```bash
git clone https://github.com/gehad3sam/BUkhari_RAG.git
cd BUkhari_RAG
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Add your API keys to `.env`:**

```ini
QDRANT_URL="https://your-cluster-url.qdrant.io"
QDRANT_API_KEY="your_qdrant_api_key"
COHERE_API_KEY="your_cohere_api_key"
GEMINI_API_KEY="your_gemini_api_key"
```

**4. Launch the application:**

```bash
streamlit run app.py
```

---

## 📁 هيكل ملفات المشروع (Project Structure)

```text
BUkhari_RAG/
├── app.py              # الكود الرئيسي لتطبيق Streamlit ومحرك الـ RAG
├── requirements.txt    # الحزم والمكتبات البرمجية المطلوبة للتشغيل
├── .gitignore          # استبعاد ملفات البيئة الافتراضية (.venv) ومفاتيح (.env)
└── README.md           # دليل التوثيق وطريقة التشغيل
```

---

## ⚠️ إخلاء مسؤولية (Disclaimer)

يعتمد هذا التطبيق على نماذج الذكاء الاصطناعي في تلخيص وصياغة الإجابات من الأحاديث المسترجعة، وقد يعتري صياغته شيء من الخطأ؛ يظل النص العربي الأصلي للحديث المرفق في قسم **«الأحاديث التي تم الاعتماد عليها»** هو المرجع النهائي والمعتمد.