from fastapi import FastAPI, HTTPException
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
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = FastAPI()

# 1. LLM yükle
try:
    llm = Llama(
        model_path="../models/llama2/llama-2-13b.Q5_K_M.gguf",
        n_ctx=2048,
        n_threads=4
    )
    logging.info("LLM modeli başarıyla yüklendi.")
except Exception as e:
    logging.error(f"LLM modeli yüklenemedi: {e}")
    raise

# 2. FAISS ve metinleri yükle
try:
    index = faiss.read_index("../faiss_db/infra_docs.index")
    with open("../faiss_db/infra_texts.pkl", "rb") as f:
        data = pickle.load(f)
    texts = data["texts"]
    logging.info("FAISS index ve metinler başarıyla yüklendi.")
except Exception as e:
    logging.error(f"FAISS verileri yüklenemedi: {e}")
    raise

# 3. Embedding modeli yükle
try:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info("Embedding modeli yüklendi.")
except Exception as e:
    logging.error(f"Embedding modeli yüklenemedi: {e}")
    raise

# 4. API veri modeli
class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    logging.info("Yeni istek alındı.")
    
    if not query.question.strip():
        logging.warning("Boş soru alındı.")
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")

    try:
        # Soru için embedding üret
        question_embedding = embed_model.encode([query.question])
        logging.info("Soru için embedding oluşturuldu.")

        # En yakın 3 metni getir
        D, I = index.search(np.array(question_embedding), k=3)
        context = "\n".join([texts[i] for i in I[0]])
        logging.info(f"Benzer metinler alındı: {I[0].tolist()}")

        # Prompt hazırla
        prompt = f"""
Aşağıda sistemlerle ilgili bazı bilgiler verilmiştir:
{context}

Soru: {query.question}
Cevap:"""

        # Modeli çalıştır
        response = llm(prompt, max_tokens=200, temperature=0.7, top_p=0.95, stop=["Soru:"])
        answer = response["choices"][0]["text"].strip()
        logging.info("Model cevabı başarıyla üretildi.")

        return {"answer": answer}
    
    except Exception as e:
        logging.error(f"İşlem sırasında hata oluştu: {e}")
        raise HTTPException(status_code=500, detail="Cevap üretilemedi.")
