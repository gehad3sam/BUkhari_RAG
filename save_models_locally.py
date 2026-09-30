from huggingface_hub import snapshot_download

print("⏳ Downloading BGE-M3 directly into ./models/bge-m3...")
snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir="./models/bge-m3",
    local_dir_use_symlinks=False
)

print("⏳ Downloading Reranker directly into ./models/bge-reranker...")
snapshot_download(
    repo_id="BAAI/bge-reranker-v2-m3",
    local_dir="./models/bge-reranker",
    local_dir_use_symlinks=False
)

print("🎉 Models saved locally inside your project folder!")