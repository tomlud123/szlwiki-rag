import time
import torch
from langchain_huggingface import HuggingFaceEmbeddings

# 🔧 Uruchamianie na: CPU
# ⏱️  Czas: 76.10 sekundy
#
# 🔧 Uruchamianie na: CUDA
# ⏱️  Czas: 7.69 sekundy
# 🧠 CUDA dostępna: True
# 📊 Używana karta: NVIDIA GeForce RTX 5070
# 💾 Zużycie pamięci: 2143.9 MB

# losowe dane do embedowania
N = 10_000
texts = ["To jest testowy tekst nr {}".format(i) for i in range(N)]

def run_benchmark(device: str):
    print(f"\n🔧 Uruchamianie na: {device.upper()}")
    model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 🔹 Pomiar czasu
    start = time.perf_counter()
    _ = model.embed_documents(texts)
    end = time.perf_counter()

    duration = end - start
    print(f"⏱️  Czas: {duration:.2f} sekundy")

    if device == "cuda":
        # 🔹 Pokaż dodatkowe informacje GPU
        print(f"🧠 CUDA dostępna: {torch.cuda.is_available()}")
        print(f"📊 Używana karta: {torch.cuda.get_device_name(0)}")
        print(f"💾 Zużycie pamięci: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

if __name__ == "__main__":
    run_benchmark("cpu")
    run_benchmark("cuda")
