import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
vector_store = None

def read_pdf(path):
    reader = PdfReader(path)
    return "\n".join(p.extract_text() or "" for p in reader.pages)

@app.route("/", methods=["GET", "POST"])
def index():
    global vector_store
    answer = None
    question = None

    if request.method == "POST":
        action = request.form.get("action")
        if action == "upload":
            file = request.files["file"]
            path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
            file.save(path)

            text = read_pdf(path)
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=150
            ).split_text(text)

            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vector_store = FAISS.from_texts(chunks, embeddings)

        elif action == "ask":
            question = request.form["question"]
            if vector_store is None:
                answer = "Please upload a PDF first."
            else:
                docs = vector_store.similarity_search(question, k=4)
                context = "\n\n".join(d.page_content for d in docs)

                prompt = f"""Answer the question using the uploaded document context.
If the context is insufficient, say what the document does not contain.
You may use general LLM knowledge only after clearly separating it from the document answer.

CONTEXT:
{context}

QUESTION: {question}
"""
                llm = ChatOllama(model=MODEL)
                answer = llm.invoke(prompt).content

    return render_template("index.html", answer=answer, question=question)

if __name__ == "__main__":
    app.run(debug=True)
