import os
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from groq import Groq
import cohere
# ==========================================
# 1. PAGE CONFIGURATION & RTL STYLING
# ==========================================
st.set_page_config(page_title="الباحث في صحيح البخاري", page_icon="🕌", layout="centered")
rtl_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');

    /* Apply font to everything EXCEPT Streamlit's Material Icons */
    *:not(.material-symbols-rounded):not(.icon):not(svg):not(i) { 
        font-family: 'Tajawal', sans-serif !important; 
    }

    /* Apply native RTL direction to the main app container */
    .block-container {
        direction: rtl !important;
    }
    /* Explicitly restore Streamlit's UI icons */
    .material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important;
    }

    /* Ensure all text elements inherit the right alignment */
    p, div, span, li, h1, h2, h3, h4, h5, h6, textarea, input {
        text-align: right !important;
    }

    /* Keep the chat avatars neat */
    div[data-testid="stChatMessage"] {
        direction: rtl !important;
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


from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# # Define local folder paths
# BGE_M3_LOCAL_PATH = "./models/bge-m3"
# RERANKER_LOCAL_PATH = "./models/bge-reranker"
# @st.cache_resource(show_spinner="جاري تحميل النماذج من القرص المحلي...")
# def load_ai_models():
#     # Pass the local path instead of "BAAI/bge-m3"
#     embeddings = HuggingFaceEmbeddings(
#         model_name=BGE_M3_LOCAL_PATH,
#         model_kwargs={
#             'device': 'cpu',
#             'local_files_only': True  # Strictly prevents online requests
#         },
#         encode_kwargs={'normalize_embeddings': True}
#     )
#
#     # Pass the local path instead of "BAAI/bge-reranker-v2-m3"
#     reranker = CrossEncoder(
#         RERANKER_LOCAL_PATH,
#         max_length=256,
#         device='cpu',
#         automodel_args={'local_files_only': True}  # Strictly prevents online requests
#     )
#
#     qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
#     groq = Groq(api_key=GROQ_API_KEY)
#
#     return embeddings, reranker, qdrant, groq

# for deploying the Project on to Hugging Face
# @st.cache_resource(show_spinner="جاري تحميل نماذج الذكاء الاصطناعي...")
# def load_ai_models():
#     # Load directly from HF Hub (Cloud server will cache this instantly)
#     embeddings = HuggingFaceEmbeddings(
#         model_name="BAAI/bge-m3",
#         model_kwargs={'device': 'cpu'},
#         encode_kwargs={'normalize_embeddings': True}
#     )
#
#     reranker = CrossEncoder(
#         "BAAI/bge-reranker-v2-m3",
#         max_length=256,
#         device='cpu'
#     )
#
#     qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
#     groq = Groq(api_key=GROQ_API_KEY)
#
#     return embeddings, reranker, qdrant, groq


# for deploying in streamlit
from langchain_huggingface import HuggingFaceEndpointEmbeddings
@st.cache_resource(show_spinner="جاري تحميل نماذج الذكاء الاصطناعي...")
def load_ai_models():
    # --- NEW: Cloud API Embeddings ---
    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-m3",
        task="feature-extraction",
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )

    qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

    return embeddings, None, qdrant, groq  # We return None for reranker since we use Cohere directly now

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
    avatar_icon = "🙋‍♂️" if msg["role"] == "user" else "🕌"
    with st.chat_message(msg["role"], avatar = avatar_icon):
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
    with st.chat_message("user", avatar="🙋‍♂️"):
        st.write(user_query)

    # 2. Process Assistant Response
    with st.chat_message("assistant", avatar="🕌"):
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

            SIMILARITY_THRESHOLD = 0.40
            # candidate_chunks = search_results.points
            candidate_chunks = [point for point in search_results.points if point.score >= SIMILARITY_THRESHOLD]
            if not candidate_chunks:
                status_text.empty()  # Clear loading text
                out_of_scope_msg = "عذراً،لا يمكنني إيجاد جواب هذا السؤال من الأحاديث النبوية المتوفرة في صحيح البخاري."
                st.write(out_of_scope_msg)

                # Save to history so it doesn't disappear
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": out_of_scope_msg
                })

                st.stop()  # 🛑 Completely halts the script here (Reranker & LLM are skipped)



            # --- STAGE 2: RERANKING ---
            # status_text.info("🎯 جاري الفرز الدقيق للأحاديث (Reranker)...")
            # pairs = [[user_query, point.payload['original_text']] for point in candidate_chunks]
            # rerank_scores = reranker_model.predict(pairs)
            #
            # reranked_results = []
            # for score, point in zip(rerank_scores, candidate_chunks):
            #     reranked_results.append({
            #         "score": float(score),
            #         "hadith_number": point.payload['hadith_number'],
            #         "book_title": point.payload['book_title'],
            #         "text": point.payload['original_text']
            #     })
            #
            # reranked_results.sort(key=lambda x: x["score"], reverse=True)
            #
            # unique_results = []
            # seen_texts = set()
            # for chunk in reranked_results:
            #     if chunk['text'] not in seen_texts:
            #         unique_results.append(chunk)
            #         seen_texts.add(chunk['text'])
            #
            # top_3_chunks = unique_results[:3]

             # Add this to the top of your file

            # --- STAGE 2: RERANKING (Using Cohere API) ---
            status_text.info("🎯 جاري الفرز الدقيق للأحاديث (Cohere)...")

            # 1. Deduplicate Qdrant results first to save API tokens
            unique_candidates = []
            seen_texts = set()
            for point in candidate_chunks:
                text = point.payload['original_text']
                if text not in seen_texts:
                    unique_candidates.append(point)
                    seen_texts.add(text)

            # 2. Extract just the text strings for Cohere
            docs = [point.payload['original_text'] for point in unique_candidates]

            # 3. Call the Multilingual Reranker
            co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
            response = co.rerank(
                model="rerank-multilingual-v3.0",
                query=user_query,
                documents=docs,
                top_n=3
            )

            # 4. Map the winning results back to their full metadata
            top_3_chunks = []
            for result in response.results:
                original_point = unique_candidates[result.index]  # Match index from Cohere response
                top_3_chunks.append({
                    "score": result.relevance_score,  # Optional: save the score
                    "hadith_number": original_point.payload['hadith_number'],
                    "book_title": original_point.payload['book_title'],
                    "text": original_point.payload['original_text']
                })
            #
            unique_results = []
            seen_texts = set()
            for chunk in top_3_chunks:
                if chunk['text'] not in seen_texts:
                    unique_results.append(chunk)
                    seen_texts.add(chunk['text'])

            top_3_chunks = unique_results[:3]

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
            stream = llm_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                max_tokens=1024,
                stream = True
            )

            status_text.empty()
            def parse_groq_stream(stream_obj):
                for chunk in stream_obj:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content

            # final_answer = response.choices[0].message.content
            final_answer = st.write_stream(parse_groq_stream(stream))
            # --- STAGE 4: RENDER FINAL OUTPUT ---
            # status_text.empty()  # Clear the loading status
            # st.write(final_answer)  # Print LLM response

            # Render the source expander
            with st.expander("الأحاديث التي تم الاعتماد عليها (اضغط للتوسيع)"):
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