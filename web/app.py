from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

FASTAPI_CHAT_URL = "http://localhost:8000/chat"  # FastAPI chat endpoint

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Lütfen bir soru yazın."}), 400

    try:
        resp = requests.post(FASTAPI_CHAT_URL, json={"question": question}, timeout=15)
        resp.raise_for_status()
        answer = resp.json().get("answer", "Cevap alınamadı.")
    except requests.RequestException as e:
        answer = f"Sunucu ile bağlantı kurulamadı: {str(e)}"
    except Exception as e:
        answer = f"Bilinmeyen hata oluştu: {str(e)}"

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
