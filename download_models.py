from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

print("⏳ Downloading BGE-M3 Embeddings (approx. 2.2 GB)...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ Embeddings downloaded!\n")

print("⏳ Downloading BGE Reranker (approx. 2.2 GB)...")
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cpu')
print("✅ Reranker downloaded!\n")

print("🎉 All models cached locally. You can now start Streamlit!")