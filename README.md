<div align="center">

🕌 الباحث الذكي في مختصر صحيح البخاري

Classical Arabic Hadith RAG System (Mukhtasar Sahih al-Bukhari)

🇬🇧 English Documentation | 🇸🇦 التوثيق باللغة العربية

</div>

<a name="-english-documentation"></a>

🇬🇧 English Documentation

📖 Overview

AskBukhari is a production-grade Retrieval-Augmented Generation (RAG) web application engineered specifically for Classical Arabic Hadith literature — based on Al-Tajrid Al-Sarih li-Ahadith Al-Jami' Al-Sahih (Mukhtasar Sahih al-Bukhari by Imam Al-Zubaidi, comprising 2,200+ authenticated hadiths).

Unlike standard RAG demos built on modern prose, classical Arabic prophetic traditions feature dense syntax, attached pronouns, classical homonyms, and multiple historical events narrated within a single passage. A single misresolved pronoun or conflated event can invert a legal ruling or misattribute an action. This system solves those challenges through a Two-Stage Retrieval Pipeline paired with Strict Linguistic & Semantic Guardrails.

🔗 Try the Live Application: https://askbukhari.streamlit.app/

⚙️ System Architecture & Pipeline

User Query (Modern / Dialect / Classical Arabic)
   │
   ├─► Greeting Regex Filter ──► Instant Polite Greeting (Zero API Cost)
   │
   ▼
[Stage 1: Dense Vector Retrieval]
   • Model: Cohere `embed-multilingual-v3.0`
   • Database: Qdrant Cloud (`sahih_bukhari_with_cohere`, Top-K = 15)
   • Guardrail: Similarity Threshold >= 0.42 (Early exit for out-of-scope queries to save tokens & latency)
   │
   ▼
[Stage 2: Deduplication & Contextual Cross-Encoder Reranking]
   • Deduplication: Removes duplicate hadith texts across chunks
   • Contextual Enrichment: Prepends `Book Title` to `Hadith Text` (`كتاب: {book_title} | {original_text}`)
   • Model: Cohere `rerank-multilingual-v3.0` (Top-N = up to 7, Relevance Score >= 0.10 with min-3 fallback)
   │
   ▼
[Stage 3: Low-Latency Streaming Generation]
   • Model: Google `gemini-3.5-flash-lite` (`temperature = 0.0`, `ThinkingLevel.MINIMAL`)
   • Guardrails: Pronoun Resolution, Silent Entity-Role Filtering, Event Boundary Disambiguation, Dual Citation
   • Output: Real-time token streaming + Expandable verbatim source hadiths


🧠 Key Engineering Challenges & Solutions

Event Boundary Disambiguation (فصل الأحداث والغزوات المتداخلة):

Challenge: Several long hadiths mention multiple battles in the same text (e.g., mentioning veterans of the Battle of Badr inside the narrative of the Battle of Tabuk in the hadith of Ka'b ibn Malik).

Solution: Engineered strict temporal/spatial boundary rules in the system instruction after 6+ iterative evaluation & error-analysis rounds, ensuring every ruling and companion is bound strictly to their respective historical event.

Silent Entity-Role Resolution (التفكيك الدلالي للأدوار):

Challenge: In multi-page narratives (such as Hadith al-Ifk), numerous figures appear as narrators, witnesses, advisors, victims, or direct actors. Naive LLMs frequently conflate adjacent names.

Solution: Enforced silent syntactic and semantic role filtering so the model isolates only the direct grammatical actors matching the user's question without hallucinating or listing excluded figures.

Bridging the Dialect/Modern Query & Classical Text Gap (Latency vs. Accuracy Trade-off):

Challenge: Users often ask questions in informal or modern standard Arabic while the corpus uses classical wording.

Solution: Experimented with LLM-based pre-retrieval Query Expansion, which doubled API latency and consumed rate limits without proportional accuracy gains. Replaced it with Book-Title Contextualized Multilingual Reranking (Cohere v3) + ThinkingLevel.MINIMAL in Gemini 3.5 Flash-Lite, cutting end-to-end latency to ~2–4 seconds.

Model Benchmarking & Adversarial Red-Teaming:

Embeddings: Migrated from local BGE-M3 (high cold-start and inference latency) to cloud-hosted Cohere embed-multilingual-v3.0.

LLMs: Benchmarked Groq-hosted models (Llama-3.3-70B, GPT-OSS-120B, and Qwen) against Gemini 3.5 Flash-Lite. While Qwen offered high raw speed, it struggled with nuanced classical Arabic pronouns; Gemini 3.5 Flash-Lite with ThinkingLevel.MINIMAL delivered superior Arabic comprehension, low latency, and resilience against Prompt Injection and false-premise queries.

Mobile-First RTL User Interface:

Custom CSS supporting Right-to-Left (RTL) typography (Tajawal font), bidirectional bracket/number isolation (unicode-bidi: isolate), responsive mobile breakpoints (< 768px), state-persistent quick-start query buttons via Streamlit callbacks, and a pinned AI disclaimer footer.

🚀 Local Installation & Setup

Clone the repository:

git clone https://github.com/gehad3sam/BUkhari_RAG.git
cd BUkhari_RAG


Install dependencies:

pip install -r requirements.txt


Configure environment variables:
Create a .env file in the project root:

QDRANT_URL="your_qdrant_cluster_url"
QDRANT_API_KEY="your_qdrant_api_key"
COHERE_API_KEY="your_cohere_api_key"
GEMINI_API_KEY="your_gemini_api_key"


Run the Streamlit app:

streamlit run app.py


<div dir="rtl">

<a name="-التوثيق-باللغة-العربية"></a>

🇸🇦 التوثيق باللغة العربية

📖 نبذة عن المشروع

تطبيق «الباحث الذكي في مختصر صحيح البخاري» هو نظام استرجاع معزز بالتوليد (RAG) صُمّم للإجابة عن الأسئلة الشرعية والحديثية والتاريخية بالاعتماد حصرياً على كتاب «التجريد الصريح لأحاديث الجامع الصحيح» للإمام الزبيدي (أكثر من 2200 حديث)، مع توثيق مزدوج ومنع صارم للهلوسة أو الاستنتاج من خارج النص.

🔗 رابط تجربة التطبيق مباشرة: https://askbukhari.streamlit.app/

⚙️ معمارية النظام (System Architecture)

تجهيز البيانات والـ Metadata:

تنظيف نصوص الأحاديث وتقسيمها، مع ربط كل حديث ببياناته الوصفية الدقيقة: (book_title اسم الكتاب، hadith_number رقم الحديث في المختصر، ورقم الحديث في صحيح البخاري الأصلي).

الاسترجاع الدلالي الأولي (Stage 1 — Dense Retrieval):

تحويل سؤال المستخدم إلى متجهات عبر نموذج Cohere (embed-multilingual-v3.0) والبحث في قاعدة بيانات Qdrant لاسترجاع أعلى 15 نتيجة مرشحة.

تطبيق فلتر تشابه مبدئي (SIMILARITY_THRESHOLD = 0.42) لرفض الأسئلة الخارجة عن النطاق مبكراً، مما يوفر استهلاك الـ API ويقلل زمن الاستجابة (Latency).

إزالة التكرار وإعادة الترتيب السياقي (Stage 2 — Deduplication & Reranking):

إزالة النصوص المكررة (Deduplication)، ثم دمج عنوان الكتاب مع متن الحديث (كتاب: ... | نص الحديث) وإرساله إلى نموذج Cohere (rerank-multilingual-v3.0) لاختيار أدق الأحاديث صلةً بالسؤال (top_n = 7 مع حد قبول relevance_score >= 0.10).

التوليد اللغوي والبث الحي (Stage 3 — Streaming Generation):

توليد الإجابة عبر نموذج Google (gemini-3.5-flash-lite) بدرجة حرارة 0.0 وإعداد ThinkingLevel.MINIMAL لضمان أسرع استجابة لحظية (Streaming)، مع إرفاق متون الأحاديث الأصلية الكاملة أسفل كل إجابة.

💡 أبرز التحديات الهندسية وكيف تم حلها

تداخل الأحداث والشخصيات في الحديث الواحد:
مثل ورود ذكر غزوة بدر داخل حديث يتحدث عن غزوة تبوك، أو تعدد الأسماء في الأحاديث الطويلة (كحديث الإفك) بين راوٍ وشاهد ومستشار وفاعل، بالإضافة إلى الألفاظ التراثية ذات المعاني الخاصة. تم التغلب على ذلك عبر أكثر من 6 جولات من التقييم المنهجي وتحليل الأخطاء (Error Analysis)، وهندسة قواعد تحليل نحوي ودلالي داخل الـ System Prompt تفصل بصمت بين الأدوار والأحداث.

الفجوة بين سؤال المستخدم بالعامية ولغة الحديث (مفاضلة الـ Latency مقابل الدقة):
تمت تجربة إضافة مرحلة Query Expansion عبر الـ LLM لتوليد مرادفات تراثية قبل البحث، لكن التجربة العملية أثبتت أنها تضاعف الـ Latency وتستهلك حدود الـ Rate Limits دون تحسن موازٍ في الدقة؛ فتم الاستغناء عنها لصالح تقوية مرحلة الـ Contextual Reranking وضبط ThinkingLevel.MINIMAL.

المقارنة بين النماذج ومقاومة هجمات الـ Prompt Injection:

في مرحلة الـ Embeddings: تم استبدال نموذج BGE-M3 بسبب بطء التحميل والاستجابة بنموذج Cohere السحابي.

في مرحلة الـ LLM: بعد مقارنة عملية بين عدة نماذج على منصة Groq (Llama-3.3-70B وGPT-OSS-120B وQwen الذي كان الأسرع لكن ظهرت فيه أخطاء في فهم الضمائر والسياق العربي)، استقر الاختيار على Gemini 3.5 Flash-Lite لدقته العالية في فهم النصوص التراثية وسرعة استجابته وصموده أمام اختبارات الـ Prompt Injection والأسئلة المفخخة.

تحسين تجربة المستخدم على الهاتف (Mobile-First RTL UI):

تصميم واجهة عربية كاملة من اليمين لليسار (RTL)، وإصلاح انقلاب الأقواس والأرقام في نهاية الأسطر (unicode-bidi: isolate)، وإضافة أزرار أسئلة سريعة تعمل عبر Session State Callbacks، مع تثبيت تنويه إخلاء المسؤولية أسفل شريط البحث.

📁 هيكل المشروع (Project Structure)

BUkhari_RAG/
├── app.py              # تطبيق Streamlit الرئيسي وخط أنابيب الـ RAG
├── requirements.txt    # المكتبات والحزم البرمجية المطلوبة
├── .gitignore          # استبعاد ملفات البيئة الافتراضية ومفاتيح .env
└── README.md           # توثيق المشروع (إنجليزي / عربي)


⚠️ إخلاء مسؤولية

يعتمد هذا التطبيق على نماذج الذكاء الاصطناعي في تلخيص وصياغة الإجابات من الأحاديث المسترجعة، وقد يعتري صياغته شيء من الخطأ؛ يظل النص العربي الأصلي للحديث المرفق في قسم «الأحاديث التي تم الاعتماد عليها» هو المرجع النهائي والمعتمد.

</div>