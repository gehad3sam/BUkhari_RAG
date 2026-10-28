import os
import re
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
# # from groq import Groq
# from openai import OpenAI

import cohere
# ==========================================
# 1. PAGE CONFIGURATION & RTL STYLING
# ==========================================
st.set_page_config(page_title="الباحث في صحيح البخاري", page_icon="🕌", layout="centered")

rtl_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');

    /* 1. Apply font globally BUT safely (Removed the aggressive * selector) */
    html, body, p, div, h1, h2, h3, h4, h5, h6, li, a, textarea, input, button { 
        font-family: 'Tajawal', sans-serif !important; 
    }

    /* 2. 🚨 PROTECT MATERIAL ICONS 🚨 */
    .material-symbols-rounded, 
    [class*="material-symbols"], 
    [class*="material-icons"],
    svg, i {
        font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
        unicode-bidi: normal !important;
        direction: ltr !important; /* Icons must remain LTR to prevent rendering bugs */
    }

    /* 3. General RTL Alignment */
    .block-container, div[data-testid="stChatMessage"] {
        direction: rtl !important;
    }

    p, div, span, li, h1, h2, h3, h4, h5, h6, textarea, input {
        text-align: right !important;
    }

    /* 4. Fix Arabic Lists */
    .stMarkdown ol, .stMarkdown ul {
        direction: rtl !important;
        text-align: right !important;
        padding-right: 2.5rem !important;
        padding-left: 0 !important;
        margin-right: 20px !important;
        margin-left: 0 !important;
    }

    /* 5. Fix Expander (Accordion) Overlap */
    details summary {
        direction: rtl !important;
        text-align: right !important;
        display: flex !important;
        align-items: center !important;
    }
    details summary > span {
        margin-left: 10px !important; /* Adds breathing room between the icon and text */
    }
    /* Fix the typing cursor (caret) behaving weirdly in the chat input */
    div[data-testid="stChatInput"] textarea {
        direction: rtl !important;
        text-align: right !important;
        caret-color: auto !important;
    }
    
    /* 6. إصلاح اتجاه الجداول (Markdown Tables) لتكون من اليمين لليسار بالكامل */
    .stMarkdown table, table {
        direction: rtl !important;
        width: 100% !important;
        border-collapse: collapse !important;
    }
    .stMarkdown th, .stMarkdown td, th, td {
        direction: rtl !important;
        text-align: right !important;
        padding: 8px 12px !important;
    }
</style>
"""
st.markdown(rtl_css, unsafe_allow_html=True)




# --- Sidebar: App Info & Tech Stack ---
with st.sidebar:
    st.title("عن التطبيق ℹ️")
    st.markdown("""
    هذا التطبيق هو محرك بحث ذكي مخصص للبحث في **صحيح البخاري** باستخدام تقنيات توليد الاسترجاع المعزز (RAG).

    **التقنيات المستخدمة:**
    - **واجهة المستخدم:** Streamlit
    - **قاعدة البيانات (Vector DB):** Qdrant
    - **محرك التوليد (LLM):** Groq
    - **التضمين والترتيب (Embeddings/Reranking):** Cohere

    ---
    **⚠️ إخلاء مسؤولية:**
    يعتمد هذا التطبيق على نماذج الذكاء الاصطناعي التي قد تخطئ أو تهلوس أحياناً في صياغة الشرح. النص العربي الأصلي للأحاديث المرفقة هو المرجع النهائي والوحيد.
    """)

st.title("🕌 الباحث الذكي في صحيح البخاري")
st.write("أهلاً بك. اطرح سؤالك وسيقوم النظام بالبحث في الأحاديث وتوليد إجابة دقيقة.")

# ==========================================
# 2. LOAD ENVIRONMENT VARIABLES & MODELS
# ==========================================
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# COLLECTION_NAME = "sahih_bukhari"
COLLECTION_NAME= "sahih_bukhari_with_cohere"# Update if you named it differently in Colab


# Remove the Hugging Face import and add Cohere


from google import genai
from google.genai import types
from langchain_cohere import CohereEmbeddings

# 1. تعريف العميل في البداية:

@st.cache_resource(show_spinner="جاري تحميل نماذج الذكاء الاصطناعي...")
def load_ai_models():
    # --- UPGRADED: Free Cohere Cloud Embeddings ---
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    return embeddings, cohere_client, qdrant, gemini_client

try:
    embedding_model, cohere_client, qdrant_client, llm_client = load_ai_models()
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
user_query = st.chat_input("اكتب سؤالك هنا... (مثال: ماهي السبع الموبقات؟)")

if user_query:
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🙋‍♂️"):
        st.write(user_query)

    # 2. Process Assistant Response
    with st.chat_message("assistant", avatar="🕌"):
        status_text = st.empty()  # Placeholder for live status updates

        try:

            GREETINGS_PATTERN = r"^(أهلا|مرحبا|سلام|السلام عليكم|كيف حالك|عامل ايه|صباح الخير|مساء الخير)"
            if re.search(GREETINGS_PATTERN, user_query.strip(), re.IGNORECASE) and len(user_query.split()) <= 4:
                bot_reply = "أهلاً بك! 👋 أنا الباحث الذكي في صحيح البخاري. كيف يمكنني مساعدتك اليوم؟"
                with st.chat_message("assistant", avatar="🕌"):
                    st.write(bot_reply)
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                st.stop()

            # --- STAGE 1: DENSE RETRIEVAL ---
            status_text.info("⏳ جاري البحث المبدئي في قاعدة البيانات...")


            query_vector = embedding_model.embed_query(user_query)

            search_results = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=15
            )

            SIMILARITY_THRESHOLD = 0.42
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
            # docs = [point.payload['original_text'] for point in unique_candidates]
            docs = [
                f"كتاب: {point.payload['book_title']} | {point.payload['original_text']}"
                for point in unique_candidates
            ]
            # 3. Call the Multilingual Reranker
            # co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
            response = cohere_client.rerank(
                model="rerank-multilingual-v3.0",
                query=user_query,
                documents=docs,
                top_n=min(7,len(docs))
            )

            top_chunks = []
            for result in response.results:
                if result.relevance_score >= 0.05 or len(top_chunks)<3:
                    original_point = unique_candidates[result.index]
                    top_chunks.append({
                        "score": result.relevance_score,
                        "hadith_number": original_point.payload['hadith_number'],
                        "book_title": original_point.payload['book_title'],
                        "text": original_point.payload['original_text']
                    })

            # --- STAGE 3: GROQ GENERATION ---
            status_text.info("✍️ جاري صياغة الإجابة...")


            context_blocks = []
            for i, chunk in enumerate(top_chunks, 1):
                context_blocks.append(
                    f"--- Context {i} ---\n"
                    f"Book: {chunk['book_title']}\n"
                    f"Hadith Number: {chunk['hadith_number']}\n"
                    f"Text: {chunk['text']}\n"
                )
            context_str = "\n".join(context_blocks)

            system_prompt = (
                "، أنت باحث ومحقق علمي خبير في السنة النبوية والسيرة، متخصص في كتاب «التجريد الصريح لأحاديث الجامع الصحيح» (مختصر صحيح البخاري للإمام الزبيدي).\n"
                "إذا أردت الإشارة وذكر الكتاب فقل هذا الاسم: مختصر صحيح البخاري للإمام الزبيدي واذكر اسم الكتاب كاملا في وقت الحاجة أو السؤال عنه فقط "
                "مهمتك الإجابة عن سؤال المستخدم بدقة علمية وتاريخية صارمة، بالاعتماد **حصرياً** على نصوص الأحاديث المرفقة في السياق أدناه دون أي إضافة خارجية.\n\n"

                "⚖️ الضوابط الشرعية والمنهجية الصارمة:\n"
                "1. صون مقام النبوة والصحابة وأمهات المؤمنين:\n"
                "   - التزم الأدب الرفيع عند ذكر النبي ﷺ (صلى الله عليه وسلم) والصحابة وأمهات المؤمنين (رضي الله عنهم).\n"
                "   - إياك أن تنسب إلى صحابي أو أم مؤمنين تهمة أو فعلاً شنيعاً (كالخوض في الإفك، أو الاستسلام، أو الردة) بسبب سوء فهم الضمائر أو تجاور الأسماء في النص.\n"
                "   - عند إسناد الأفعال إلى الله تعالى في النص (مثل: «فقتله الله ببدر»)، التزم بلفظ النص أو وضّح أنه قُتل في المعركة بتقدير الله على أيدي المؤمنين، ولا تقل «وقع القتل بيد الله».\n\n"

                "🧠 بروتوكول التحليل النحوي والتفكيكي (يُطبق ذهنياً بصمت قبل الصياغة):\n"
                "2. ضبط الضمائر والفاعل والمفعول (Pronoun & Syntax Resolution):\n"
                "   - حلّل الجمل العربية نحوياً بدقة قبل التلخيص: حدّد على من يعود كل ضمير متصل أو مستتر (مثل: «فهزموهم»، «فأقبلوا»، «قالا مثل ما قلت»).\n"
                "   - ميّز بين النداء والتحذير وبين أسماء الأعلام والقبائل (مثال: «الغنيمةَ أي قوم» هو نداء لجمع الغنائم وليس اسم قبيلة!).\n"
                "   - انتبه لعطف البيان والبدل والكنى (مثال: «أم مسطح بنت أبي رهم» هي امرأة واحدة وليست شخصيتين).\n\n"

                "3. التفكيك الدلالي للأدوار والفلترة الصامتة (Silent Entity-Role Filtering):\n"
                "   - عند السؤال عن فاعل حدث معين (مثل: من خاض، من تخلف، من قَتل)، اذكر في إجابتك **فقط** الشخصيات التي مارست هذا الفعل صراحة.\n"
                "   - افصل ذهنياً وبصمت بين (الفاعل المباشر) وبين غيره (كالراوي، أو المعتدى عليه، أو المستشار، أو الشاهد)، **ويُمنع منعاً باتاً إدراج غير الفاعلين في الإجابة أو الجدول، كما يُمنع ذكر أسمائهم لتبرير سبب استبعادهم أو شرح أدوارهم التي لم يسأل عنها المستخدم** (فمثلاً عند السؤال عمن خاض في الإفك: لا تذكر عائشة ولا النبي ﷺ ولا علياً ولا أسامة إطلاقاً، ولا تكتب ملاحظة تشرح لماذا لم تدرجهم).\n\n"

                "4. التفكيك الزمني والمكاني للأحداث (Event Boundary Disambiguation):\n"
                "   - إذا ذكر الحديث حدثين (مثل: غزوة تبوك وغزوة بدر في حديث كعب بن مالك، أو غزوة أحد وغزوة بدر في حديث البراء)، فاربط كل حكم وكل شخصية بالحدث الخاص بها بدقة.\n"
                "   - لا تسحب حكم حدث على آخر، ولا تذكر شخصيات الحدث الآخر لتبرير استبعادها (اذكر فقط من ينطبق عليه سؤال المستخدم مباشرة).\n\n"

                "5. التقييد الصارم بالنص والتوثيق المزدوج (Strict Grounding & Citation):\n"
                "   - لا تستنتج أي معلومة أو عدد أو اسم غير مذكور صراحة في السياق المرفق.\n"
                "   - إذا تضمن نص الحديث إشارة من الإمام الزبيدي لاختصار القصة (مثل: «تقدم»، أو «وفي الحديث قصة هذا مضمون الحديث منها»)، فنبه المستخدم بأمانة أن القصة مختصرة في هذا الموضع.\n"
                "   - وثّق كل معلومة بذكر: (اسم الكتاب)، و(رقم الحديث في المختصر)، و(رقم البخاري الأصلي المذكور بين قوسين في نهاية الحديث).\n\n"

                "6. قاعدة التعامل مع غياب الجواب (Zero-Result Rule):\n"
                "   - إذا كانت الأحاديث المرفقة في السياق لا تحتوي على إجابة للسؤال المطروح، **لا تنشئ جدولاً لأحاديث غير ذات صلة**، بل اعتذر بوضوح في سطرين قائلاً: «لا توجد في النصوص المسترجعة من مختصر صحيح البخاري إجابة صريحة عن هذا السؤال»، واذكر فقط إن كانت هناك إشارة عابرة ذات صلة.\n\n"

                "📊 قواعد التنسيق والاقتصاد في الكلمات (مهم جداً):\n"
                "7. الإجابة الصافية والمباشرة:\n"
                "   - قدّم الإجابة الصحيحة فقط؛ لا تذكر الاحتمالات الخاطئة، ولا تشرح لماذا استبعدت شخصيات أو أحاديث أخرى غير مطابقة للسؤال، ولا تكتب قوائم بالأحاديث المستبعدة.\n"
                "   - كن موجزاً ومباشراً ومركّزاً دون إطالة أو تكرار.\n"
                "   - **لا تستخدم الجداول (Markdown Tables) بشكل مفرط:** استخدم السرد النقطي المنظم (Bullet Points) مع العناوين الفرعية الواضحة كأصل عام للإجابة، ولا تستخدم الجدول إلا إذا كان السؤال يطلب مقارنة مباشرة بين عدة شخصيات أو أحداث متقابلة (بحد أقصى جدول واحد مختصر يضم المطلوب فقط).\n"
                "   - يُمنع استخدام أي وسوم HTML مثل <br> نهائياً."
            )



            full_prompt = f"السؤال: {user_query}\n\nالأحاديث المسترجعة (السياق):\n{context_str}\n\nيرجى تقديم إجابة دقيقة:"

            response_stream = llm_client.models.generate_content_stream(
                model="gemini-3.5-flash-lite",  # أو gemini-2.5-flash
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=2048,
                    system_instruction=system_prompt,
                    thinking_config=types.ThinkingConfig(
                        # Replaces thinking_budget=0 for newer models
                        thinking_level=types.ThinkingLevel.MINIMAL
                    )
                )
            )


            def stream_gemini(response):
                first_chunk = True
                for chunk in response:
                    if chunk.text:
                        if first_chunk:
                            status_text.empty()  # إخفاء "جاري صياغة الإجابة..." فور بدء الكتابة!
                            first_chunk = False
                        yield chunk.text

            final_answer = st.write_stream(stream_gemini(response_stream))

            # Render the source expander
            with st.expander("الأحاديث التي تم الاعتماد عليها (اضغط للتوسيع)📚"):
                for source in top_chunks:
                    st.markdown(
                        f"**الكتاب:** {source['book_title']} | **حديث رقم:** {source['hadith_number']}\n\n{source['text']}")
                    st.divider()

            # Save to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "sources": top_chunks
            })

        except Exception as e:
            status_text.empty()
            st.error(f"حدث خطأ أثناء معالجة الطلب: {e}")