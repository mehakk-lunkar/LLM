from flask import Flask, render_template, request
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        question = request.form["question"]
        answer = ask_groq(question)
    return render_template("index.html", answer=answer)
def ask_groq(prompt):
    chat_completion = client.chat.completions.create(
        model="llama3-70b-8192", 
        messages=[
            {"role": "system", "content": "You are a helpful and descriptive AI assistant for a question-answering system."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800,
        top_p=1,
        stop=None
    )
    return chat_completion.choices[0].message.content

if __name__ == "__main__":
    app.run(debug=True)
