from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_chunks():
    # 1. Wczytaj dane
    print("Rozpoczęto wczytywanie danych")
    ds = load_dataset("ipipan/silesian-wikipedia-clean-20230901", split="train")
    texts = [row["text"] for row in ds]

    # 2. Chunkowanie
    print("Rozpoczęto chunkowanie")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=50,
        separators=["\n\n",  # 1. podwójny Enter  → akapit
                    "\n",  # 2. pojedynczy Enter → wiersz
                    ". ",  # 3. kropka + spacja  → zdanie
                    " ",  # 4. spacja           → słowo
                    ""],  # 5. brak separatora  → znak
        keep_separator="end"    )

    # Zchunkuj każdy tekst osobno
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))

    print("Zakończono chunkowanie")

    return chunks  # to już lista stringów


if __name__ == "__main__":
    docs = get_chunks()
    print(f"Zchunkowanych fragmentów: {len(docs)}")
    print("Przykład:\n", docs[0][:500])
