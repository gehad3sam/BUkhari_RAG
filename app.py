import os
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import cohere
from google import genai
from langchain_cohere import CohereEmbeddings

from src.retriever import retrieve_candidates, rerank_candidates
from src.generator import is_greeting, generate_answer_stream

# ==========================================
# 1. PAGE CONFIGURATION & ADAPTIVE ISLAMIC UI
# ==========================================
st.set_page_config(
    page_title="الباحث في صحيح البخاري",
    page_icon="🕌",
    layout="centered",
    initial_sidebar_state="collapsed",
)

adaptive_islamic_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Tajawal:wght@400;500;700&display=swap');

    /* 1. Global Arabic Typography */
    html, body, p, div, h1, h2, h3, h4, h5, h6, li, a, textarea, input, button { 
        font-family: 'Tajawal', sans-serif !important; 
    }

    /* 2. Protect Material Icons from font override & RTL flipping */
    .material-symbols-rounded, 
    [class*="material-symbols"], 
    [class*="material-icons"],
    svg, i {
        font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
        unicode-bidi: normal !important;
        direction: ltr !important;
    }

    /* 3. General RTL Alignment */
    .block-container, div[data-testid="stChatMessage"] {
        direction: rtl !important;
    }

    p, div, span, li, h1, h2, h3, h4, h5, h6, textarea, input {
        text-align: right !important;
    }

    .stMarkdown p, .stMarkdown li {
        unicode-bidi: isolate !important;
        line-height: 1.8 !important;
    }

    /* 4. Adaptive Modern Islamic Hero Banner (Adapts to Light & Dark seamlessly) */
    .islamic-header {
        background: rgba(197, 155, 39, 0.08);
        border: 1px solid rgba(197, 155, 39, 0.35);
        border-right: 4px solid #C59B27;
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(8px);
    }
    .bismillah {
        font-family: 'Amiri', serif !important;
        color: #C59B27 !important;
        font-size: 1.25rem;
        text-align: center !important;
        margin-bottom: 0.35rem;
        letter-spacing: 0.5px;
        font-weight: 700;
    }
    .islamic-title {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
        text-align: center !important;
        color: inherit !important;
    }
    .islamic-subtitle {
        font-size: 0.92rem;
        margin: 0;
        text-align: center !important;
        opacity: 0.8;
        color: inherit !important;
    }

    /* 5. Adaptive Gold-Trimmed Quick Question Buttons */
    div[data-testid="stButton"] > button {
        direction: rtl !important;
        border: 1px solid rgba(197, 155, 39, 0.35) !important;
        border-radius: 12px !important;
        padding: 0.5rem 0.85rem !important;
        transition: all 0.22s ease !important;
        background: rgba(197, 155, 39, 0.06) !important;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #C59B27 !important;
        background: rgba(197, 155, 39, 0.16) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(197, 155, 39, 0.2) !important;
    }
    div[data-testid="stButton"] > button p {
        text-align: center !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
        color: inherit !important;
    }

    /* 6. Elegant Chat Bubbles & Hadith Source Expanders */
    div[data-testid="stChatMessage"] {
        border-radius: 14px !important;
        border: 1px solid rgba(197, 155, 39, 0.16) !important;
        padding: 0.9rem 1rem !important;
        margin-bottom: 0.75rem !important;
    }

    div[data-testid="stExpander"] {
        width: 100% !important;
        direction: rtl !important;
        border: 1px solid rgba(197, 155, 39, 0.28) !important;
        border-right: 3px solid #C59B27 !important;
        border-radius: 10px !important;
        margin-top: 0.6rem !important;
    }
    details summary {
        direction: rtl !important;
        text-align: right !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
    details summary > span {
        margin-left: 6px !important;
        color: #C59B27 !important;
        font-weight: 600 !important;
    }

    /* 7. Fix Arabic Bullet Lists */
    .stMarkdown ol, .stMarkdown ul {
        direction: rtl !important;
        text-align: right !important;
        padding-right: 1.25rem !important;
        padding-left: 0 !important;
        margin-right: 0.25rem !important;
        margin-left: 0 !important;
    }
    .stMarkdown li {
        margin-bottom: 0.35rem !important;
    }

    /* 8. Chat Input & Pinned Disclaimer */
    div[data-testid="stChatInput"] textarea {
        direction: rtl !important;
        text-align: right !important;
        caret-color: #C59B27 !important;
    }

    div[data-testid="stChatInput"]::after {
        content: "⚠️ الذكاء الاصطناعي قد يخطئ في الاستنباط أو التلخيص؛ يظل نص الحديث الأصلي المرفق هو المرجع المعتمد.";
        display: block;
        text-align: center !important;
        direction: rtl !important;
        font-size: 0.74rem;
        opacity: 0.72;
        margin-top: 6px;
        padding: 0 8px;
        line-height: 1.4;
        color: inherit !important;
    }

    /* 9. Fix Markdown Tables RTL */
    .stMarkdown table, table {
        direction: rtl !important;
        width: 100% !important;
        border-collapse: collapse !important;
        font-size: 0.92rem !important;
    }
    .stMarkdown th, .stMarkdown td, th, td {
        direction: rtl !important;
        text-align: right !important;
        padding: 8px 10px !important;
    }

    /* 10. Mobile Responsiveness (< 768px) */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.5rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        .islamic-header {
            padding: 1rem;
        }
        .islamic-title {
            font-size: 1.35rem;
        }
        div[data-testid="stChatMessage"] {
            padding: 0.65rem 0.55rem !important;
            gap: 0.5rem !important;
        }
        .stMarkdown ol, .stMarkdown ul {
            padding-right: 1rem !important;
            margin-right: 0 !important;
        }
        div[data-testid="stChatInput"] textarea {
            font-size: 0.9rem !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder {
            font-size: 0.82rem !important;
        }
        div[data-testid="stChatInput"]::after {
            font-size: 0.68rem !important;
            margin-top: 4px;
        }
    }
</style>
"""
st.markdown(adaptive_islamic_css, unsafe_allow_html=True)

with st.sidebar:
    st.title("عن التطبيق ℹ️")
    st.markdown("""
    هذا التطبيق هو محرك بحث ذكي مخصص للبحث في **مختصر صحيح البخاري للإمام الزبيدي** (التجريد الصريح لأحاديث الجامع الصحيح) باستخدام تقنيات الاسترجاع المعزز (RAG).

    **التقنيات المستخدمة:**
    - **واجهة المستخدم:** Streamlit
    - **قاعدة البيانات (Vector DB):** Qdrant
    - **محرك التوليد (LLM):** Gemini 3.5 Flash-Lite
    - **التضمين والترتيب (Embeddings/Reranking):** Cohere v3

    ---
    **⚠️ إخلاء مسؤولية:**
    يعتمد هذا التطبيق على نماذج الذكاء الاصطناعي التي قد تخطئ أحياناً في صياغة الشرح. النص العربي الأصلي للأحاديث المرفقة هو المرجع النهائي والوحيد.
    """)

# Hero Islamic Header Card (Adaptive)
st.markdown(
    """
    <div class="islamic-header">
        <div class="bismillah">بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ</div>
        <div class="islamic-title">🕌 الباحث الذكي في صحيح البخاري</div>
        <div class="islamic-subtitle">ابحث في أحاديث «التجريد الصريح لأحاديث الجامع الصحيح» للإمام الزبيدي بإجابات موثقة</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 2. LOAD ENVIRONMENT VARIABLES & AI CLIENTS
# ==========================================
load_dotenv()


@st.cache_resource(show_spinner="جاري تحميل نماذج الذكاء الاصطناعي...")
def load_ai_models():
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    qdrant = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return embeddings, cohere_client, qdrant, gemini_client


try:
    embedding_model, cohere_client, qdrant_client, llm_client = load_ai_models()
except Exception:
    st.error("⚠️ خطأ في تحميل النماذج. تأكد من صحة مفاتيح API في ملف .env")
    st.stop()

# ==========================================
# 3. SESSION STATE & QUICK-START QUESTIONS
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def set_example_query(q_text: str):
    st.session_state.pending_query = q_text


EXAMPLE_QUESTIONS = [
    "كيف بدأ نزول الوحي على النبي ﷺ في غار حراء؟",
    "من هم السبعة الذين يظلهم الله في ظله؟",
    "ما هي علامات المنافق وخصاله؟",
    "ما فضل قيام ليلة القدر وصيام رمضان؟",
]

st.markdown("##### 💡 أسئلة سريعة للتجربة:")
col1, col2 = st.columns(2)
for idx, q_item in enumerate(EXAMPLE_QUESTIONS):
    target_col = col1 if idx % 2 == 0 else col2
    target_col.button(
        q_item,
        key=f"quick_q_{idx}",
        use_container_width=True,
        on_click=set_example_query,
        args=(q_item,),
    )

st.divider()

# ==========================================
# 4. CHAT HISTORY DISPLAY
# ==========================================
for msg in st.session_state.messages:
    avatar_icon = "🙋‍♂️" if msg["role"] == "user" else "🕌"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.write(msg["content"])
        if "sources" in msg:
            with st.expander("📚 الأحاديث التي تم الاعتماد عليها (اضغط للتوسيع)"):
                for source in msg["sources"]:
                    st.markdown(
                        f"**الكتاب:** {source['book_title']} | **حديث رقم:** {source['hadith_number']}\n\n{source['text']}"
                    )
                    st.divider()

# ==========================================
# 5. MAIN CHAT CONTROLLER
# ==========================================
chat_input_query = st.chat_input("اكتب سؤالك هنا... (مثال: ما هي السبع الموبقات؟)")
user_query = chat_input_query or st.session_state.pending_query
st.session_state.pending_query = None

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🙋‍♂️"):
        st.write(user_query)

    with st.chat_message("assistant", avatar="🕌"):
        status_text = st.empty()

        try:
            if is_greeting(user_query):
                bot_reply = "أهلاً بك! 👋 أنا الباحث الذكي في صحيح البخاري. كيف يمكنني مساعدتك اليوم؟"
                st.write(bot_reply)
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                st.stop()

            status_text.info("⏳ جاري البحث المبدئي في قاعدة البيانات...")
            candidate_chunks = retrieve_candidates(user_query, embedding_model, qdrant_client)

            if not candidate_chunks:
                status_text.empty()
                out_of_scope_msg = "عذراً، لا يمكنني إيجاد جواب هذا السؤال من الأحاديث النبوية المتوفرة في صحيح البخاري."
                st.write(out_of_scope_msg)
                st.session_state.messages.append({"role": "assistant", "content": out_of_scope_msg})
                st.stop()

            status_text.info("🎯 جاري الفرز الدقيق للأحاديث (Cohere)...")
            top_chunks = rerank_candidates(user_query, candidate_chunks, cohere_client)

            status_text.info("✍️ جاري صياغة الإجابة...")
            final_answer = st.write_stream(
                generate_answer_stream(user_query, top_chunks, llm_client, status_text)
            )

            with st.expander("📚 الأحاديث التي تم الاعتماد عليها (اضغط للتوسيع)"):
                for source in top_chunks:
                    st.markdown(
                        f"**الكتاب:** {source['book_title']} | **حديث رقم:** {source['hadith_number']}\n\n{source['text']}"
                    )
                    st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "sources": top_chunks,
            })

        except Exception as e:
            status_text.empty()
            st.error(f"حدث خطأ أثناء معالجة الطلب: {e}")
