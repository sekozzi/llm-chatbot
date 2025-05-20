import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# CSV'yi oku
df = pd.read_csv("docs/ioasset.csv", encoding="ISO-8859-9")

# Gerekli kolonları seç (istersen tümünü kullanabiliriz)
selected_columns = [
    "AIM_OF_USE", "RESOURCE_TYPE", "OS_HOSTNAME", "OS_IP_ADDRESS",
    "OS_TYPE", "CONSOLE_IP", "VF_SERVICE_OWNER"
]

# Metne dönüştür
def row_to_text(row):
    try:
        return (
            f"Bu sistemin kullanım amacı: {row['AIM_OF_USE']}. "
            f"Kaynak türü: {row['RESOURCE_TYPE']}. "
            f"Hostname: {row['OS_HOSTNAME']}, IP adresi: {row['OS_IP_ADDRESS']}. "
            f"İşletim sistemi: {row['OS_TYPE']}, "
            f"Konsol IP'si: {row['CONSOLE_IP']}. "
            f"Servis Sahibi: {row['VF_SERVICE_OWNER']}. "
        )
    except Exception as e:
        return ""

docs = df[selected_columns].dropna().apply(row_to_text, axis=1).tolist()

# Embedding
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(docs, convert_to_numpy=True)

# FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Kaydet
faiss.write_index(index, "infra_docs.index")

with open("infra_texts.pkl", "wb") as f:
    pickle.dump({"texts": docs, "embeddings": embeddings}, f)

print("Indexleme tamamlandı.")
