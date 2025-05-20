from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

# Logging ayarları
logging.basicConfig(
    filename="chat_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

app = FastAPI()

# 1. LLM yükle
llm = Llama(
    model_path="../models/llama2/llama-2-13b.Q5_K_M.gguf",
    n_ctx=2048,
    n_threads=4
)

# 2. FAISS ve metinleri yükle
index = faiss.read_index("../faiss_db/infra_docs.index")
with open("../faiss_db/infra_texts.pkl", "rb") as f:
    data = pickle.load(f)
texts = data["texts"]

# 3. Embedding modeli
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. API modeli
class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    logging.info("Yeni soru alındı.")
    # Soruya embedding yap
    question_embedding = embed_model.encode([query.question])
    logging.info("Embedding işlemi tamamlandı.")
    
    # En yakın 3 sonucu al
    D, I = index.search(np.array(question_embedding), k=3)
    context = "\n".join([texts[i] for i in I[0]])
    logging.info("Embedding işlemi tamamlandı.")

    # Prompt hazırla
    prompt = f"""
Aşağıda sistemlerle ilgili bazı bilgiler verilmiştir:
{context}

Soru: {query.question}
Cevap:"""

    response = llm(prompt, max_tokens=200, temperature=0.7, top_p=0.95, stop=["Soru:"])
    return {"answer": response["choices"][0]["text"].strip()}
    logging.info("Model cevabı alındı.")
