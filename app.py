# import streamlit as st
#
# # 1. Configure the page tab
# st.set_page_config(page_title="الباحث في صحيح البخاري", page_icon="🕌", layout="centered")
#
# # 2. Inject CSS for Right-to-Left (RTL) and Custom Arabic Fonts
# rtl_css = """
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
#
#     /* Apply the font globally */
#     * {
#         font-family: 'Tajawal', sans-serif !important;
#     }
#
#     /* Force Right-to-Left alignment for all text and input elements */
#     p, div, input, label, h1, h2, h3, h4, h5, h6, textarea, span {
#         direction: rtl;
#         text-align: right;
#     }
# </style>
# """
# st.markdown(rtl_css, unsafe_allow_html=True)
#
# # 3. Build the User Interface
# st.title("🕌 الباحث الذكي في صحيح البخاري")
# st.write("أهلاً بك. اطرح سؤالك وسيقوم النظام بالبحث في الأحاديث وتوليد إجابة دقيقة.")
#
# # 4. Chat Interface Elements
# user_query = st.chat_input("اكتب سؤالك هنا... (مثال: من يدخل الجنة؟)")
#
# if user_query:
#     # Display user's question
#     with st.chat_message("user"):
#         st.write(user_query)
#
#     # Placeholder for the AI response
#     with st.chat_message("assistant"):
#         st.write("جاري البحث في قاعدة البيانات...")
#         # (This is where we will plug in your Qdrant + Groq logic!)


import os
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from groq import Groq

# ==========================================
# 1. PAGE CONFIGURATION & RTL STYLING
# ==========================================
st.set_page_config(page_title="الباحث في صحيح البخاري", page_icon="🕌", layout="centered")

rtl_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
    * { font-family: 'Tajawal', sans-serif !important; }
    p, div, input, label, h1, h2, h3, h4, h5, h6, textarea, span {
        direction: rtl;
        text-align: right;
    }
</style>
"""
st.markdown(rtl_css, unsafe_allow_html=True)

st.title("🕌 الباحث الذكي في صحيح البخاري")
st.write("أهلاً بك. اطرح سؤالك وسيقوم النظام بالبحث في الأحاديث وتوليد إجابة دقيقة.")

# ==========================================
# 2. LOAD ENVIRONMENT VARIABLES & MODELS
# ==========================================
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLLECTION_NAME = "sahih_bukhari"  # Update if you named it differently in Colab


@st.cache_resource(show_spinner="جاري تحميل نماذج الذكاء الاصطناعي... الرجاء الانتظار")
def load_ai_models():
    """Loads heavy models once and caches them in memory."""
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=256, device='cpu')
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    groq = Groq(api_key=GROQ_API_KEY)

    return embeddings, reranker, qdrant, groq


try:
    embedding_model, reranker_model, qdrant_client, llm_client = load_ai_models()
except Exception as e:
    st.error("⚠️ خطأ في تحميل النماذج. تأكد من صحة مفاتيح API في ملف .env")
    st.stop()

# ==========================================
# 3. CHAT HISTORY MANAGEMENT
# ==========================================
# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # If the assistant message has source Hadiths, display them in an expander
        if "sources" in msg:
            with st.expander("📚 المصادر المسترجعة (اضغط للتوسيع)"):
                for source in msg["sources"]:
                    st.markdown(
                        f"**الكتاب:** {source['book_title']} | **حديث رقم:** {source['hadith_number']}\n\n{source['text']}")
                    st.divider()

# ==========================================
# 4. MAIN RAG PIPELINE & CHAT LOGIC
# ==========================================
user_query = st.chat_input("اكتب سؤالك هنا... (مثال: من يدخل الجنة؟)")

if user_query:
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # 2. Process Assistant Response
    with st.chat_message("assistant"):
        status_text = st.empty()  # Placeholder for live status updates

        try:
            # --- STAGE 1: DENSE RETRIEVAL ---
            status_text.info("⏳ جاري البحث المبدئي في قاعدة البيانات...")
            query_vector = embedding_model.embed_query(user_query)

            search_results = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=6
            )
            candidate_chunks = search_results.points

            # --- STAGE 2: RERANKING ---
            status_text.info("🎯 جاري الفرز الدقيق للأحاديث (Reranker)...")
            pairs = [[user_query, point.payload['original_text']] for point in candidate_chunks]
            rerank_scores = reranker_model.predict(pairs)

            reranked_results = []
            for score, point in zip(rerank_scores, candidate_chunks):
                reranked_results.append({
                    "score": float(score),
                    "hadith_number": point.payload['hadith_number'],
                    "book_title": point.payload['book_title'],
                    "text": point.payload['original_text']
                })

            reranked_results.sort(key=lambda x: x["score"], reverse=True)
            top_3_chunks = reranked_results[:3]

            # --- STAGE 3: GROQ LLM GENERATION ---
            status_text.info("✍️ جاري صياغة الإجابة...")

            context_blocks = []
            for i, chunk in enumerate(top_3_chunks, 1):
                context_blocks.append(
                    f"--- Context {i} ---\n"
                    f"Book: {chunk['book_title']}\n"
                    f"Hadith Number: {chunk['hadith_number']}\n"
                    f"Text: {chunk['text']}\n"
                )
            context_str = "\n".join(context_blocks)

            system_prompt = (
                "أنت مساعد خبير ومحترم متخصص في صحيح البخاري. "
                "أجب على سؤال المستخدم بشكل صارم باستخدام نصوص الأحاديث الواردة في السياق أدناه فقط. "
                "يجب عليك دائماً ذكر رقم الحديث واسم الكتاب في إجابتك. "
                "إذا لم تجد الإجابة في السياق المرفق، فاذكر بوضوح أن المعلومات غير متوفرة في الأحاديث المتاحة ولا تألف من عندك."
            )

            user_prompt = f"السؤال: {user_query}\n\nالأحاديث المسترجعة (السياق):\n{context_str}\n\nيرجى تقديم إجابة دقيقة:"

            # Using the new 120B model as recommended by Groq
            response = llm_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1024
            )

            final_answer = response.choices[0].message.content

            # --- STAGE 4: RENDER FINAL OUTPUT ---
            status_text.empty()  # Clear the loading status
            st.write(final_answer)  # Print LLM response

            # Render the source expander
            with st.expander("📚 الأحاديث التي تم الاعتماد عليها (اضغط للتوسيع)"):
                for source in top_3_chunks:
                    st.markdown(
                        f"**الكتاب:** {source['book_title']} | **حديث رقم:** {source['hadith_number']}\n\n{source['text']}")
                    st.divider()

            # Save to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "sources": top_3_chunks
            })

        except Exception as e:
            status_text.empty()
            st.error(f"حدث خطأ أثناء معالجة الطلب: {e}")