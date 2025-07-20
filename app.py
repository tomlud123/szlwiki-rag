from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def use():
    # 1. Utwórz embedder oparty na modelu E5
    embedder = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cuda"},  # Użyj GPU
        encode_kwargs={"normalize_embeddings": True},  # Ważne dla metryk podobieństwa (dot product ≈ cosine)
    )

    # 2. Załaduj wcześniej zbudowany indeks FAISS z dysku
    vectorstore = FAISS.load_local(
        "src/data/faiss_index",
        embeddings=embedder,
        allow_dangerous_deserialization=True
    )

    # 3. Sformułuj zapytanie i wygeneruj jego embedding
    query = "Najbywitniejsi ludzie świata"
    embedded_query = embedder.embed_query("query: " + query)

    # 4. Wyszukaj najbliższe dokumenty w przestrzeni wektorowej
    results = vectorstore.similarity_search_by_vector(embedded_query)

    # 5. Wyświetl wyniki
    for i, r in enumerate(results):
        print(f"\nWynik {i + 1}:\n{r.page_content}")

if __name__ == "__main__":
    use()
