from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ingest import get_chunks
import os

# 1. Wczytaj zchunkowane dane (importowane z ingest.py)
docs = get_chunks()

# 2. Dodaj prefix "passage: " do każdego tekstu
docs = [f"passage: {doc}" for doc in docs]

# 3. Model embedujący
print("Rozpoczęto przygotowanie modelu embedującego")
embedder = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    # model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

# 3. Zbuduj indeks FAISS
print("Rozpoczęto budowę indeksu FAISS")
vectorstore = FAISS.from_texts(docs, embedding=embedder)

# 4. Zapisz indeks na dysk
print("Rozpoczęto zapis indeksu na dysk")
os.makedirs("data", exist_ok=True)
vectorstore.save_local("data/faiss_index")