from typing import List, Dict, Tuple

import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---- Configuration ---------------------------------------------------------
INDEX_PATH = "src/data/faiss_index"  # adapt if your index lives elsewhere
MODEL_NAME = "intfloat/multilingual-e5-large"  # the same model that built the index
DEVICE = "cuda"
TOP_K_DEFAULT = 3
TOP_K_MAX = 10

# ---- Load embedder & vectorstore -------------------------------------------
print("Loading embeddings model …")
embedder = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},  # important for dot‑product ≈ cosine
)

print("Loading FAISS index from", INDEX_PATH)
vectorstore = FAISS.load_local(
    INDEX_PATH,
    embeddings=embedder,
    allow_dangerous_deserialization=True,  # set to False once you trust the index
)

# ---- Chatbot callback ------------------------------------------------------

def respond(message: str, history: List[Dict[str, str]], top_k: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    retriever_with_k = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever_with_k.invoke(message)
    answer = "\n\n---\n\n".join(doc.page_content for doc in docs)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ]
    return history, history

# ---- Gradio UI -------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="Silesian Wikipedia RAG chatbot") as demoUI:
        gr.Markdown("# Silesian Wikipedia semantic search")
        gr.Markdown(
            "Zadaj pytanie, model zwróci fragmenty artykułów śląskiej Wikipedii najlepiej pasujące do twojego zapytania."
        )

        chatbot = gr.Chatbot(height=600, type='messages')
        with gr.Row():
            txt = gr.Textbox(placeholder="Np. *Typowe śląskie jedzenie*", show_label=False, scale=4)
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=TOP_K_MAX,
                value=TOP_K_DEFAULT,
                step=1,
                label="Top‑k",
                scale=1,
            )
        txt.submit(respond, [txt, chatbot, top_k_slider], [chatbot, chatbot])
        txt.submit(lambda : "", None, [txt])  # clear textbox after submit
    return demoUI


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
