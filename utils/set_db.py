import json

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

with open("/home/johannesm/qtl_tmp/full_docs.txt") as file:
    data = file.read()
    docs = json.loads(data)


db = Chroma(
    embedding_function=HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B", model_kwargs={"trust_remote_code": True}
    ),
    persist_directory="/home/johannesm/qtl_tmp/full_chroma_db",
)

chunk_size = 1

for i in tqdm(range(0, len(docs) + 1, chunk_size)):
    chunk = docs[i : i + chunk_size]
    metadatas = [{"source": f"Document {ind}"} for ind in range(i, i + len(chunk))]
    db.add_texts(texts=chunk, metadatas=metadatas)

db.persist()
