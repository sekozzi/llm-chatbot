from flask import Flask, request, render_template, session, redirect, url_for
import requests
from datetime import datetime

app = Flask(__name__)
app.secret_key = "cok-gizli-fena"

API_URL = "http://localhost:8000/chat"

@app.route("/", methods=["GET", "POST"])
def chat():
    if "history" not in session:
        session["history"] = []

    if request.method == "POST":
        question = request.form["question"]
        
        try:
            response = requests.post(API_URL, json={"question": question})
            answer = response.json().get("answer","Cevap alınamadı.")
        except Exception as e:
            answer = f"Sunucu hatası: {e}"

        session["history"].append({
            "question": question,
            "answer": answer,
            "time": datetime.now().strftime("%H:%M")
        })
        session.modified = True

    return render_template("chat.html", history=session["history"])

@app.route("/clear")
def clear():
    session.pop("history",None)
    return redirect(url_for("chat"))

if __name__ == "__main__":
    app.run(debug=True)
