from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn
import os
import logging

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# Yapılandırma değerleri
MODEL_PATH = os.environ.get("MODEL_PATH", "../models/llama2/llama-2-13b-chat-hf.Q5_K_M.gguf")
N_CTX = int(os.environ.get("N_CTX", 2048))
N_THREADS = int(os.environ.get("N_THREADS", 4))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.8))
TOP_P = float(os.environ.get("TOP_P", 0.95))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 2000))

# Modeli yükleme
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS
    )
    logging.info(f"Model yüklendi: {MODEL_PATH}")
except Exception as e:
    logging.error(f"Model yüklenirken hata oluştu: {e}")
    raise HTTPException(status_code=500, detail="Model yüklenirken hata oluştu")

# Sorgu modeli
class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    try:
        prompt = f"Soru: {query.question}\nCevap:"
        response = llm(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, stop=["Soru:"])
        answer = response["choices"][0]["text"].strip()
        logging.info(f"Soru: {query.question}, Cevap: {answer}")
        return {"answer": answer}
    except Exception as e:
        logging.error(f"Soru işlenirken hata oluştu: {e}")
        raise HTTPException(status_code=500, detail="Soru işlenirken hata oluştu")
