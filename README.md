# Silesian Wikipedia based RAG

Embeddings and Semantic Search Engine

Tech stack:
* Dataset https://huggingface.co/datasets/ipipan/silesian-wikipedia-clean-20230901
* Embedder https://huggingface.co/intfloat/multilingual-e5-large
* FAISS
* LangChain

TODO:
* config.yaml, include: chunk_size, chunk_overlap, index_name, device cuda/cpu, adapt code
* avoid unsafe deserialization of vectorstore
* optimizations
* translate to english
* add demo.ipynb/gradio
* add connection to LLM
