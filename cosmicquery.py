from flask import Flask, request, render_template, redirect, url_for, flash
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("G API key"))  

app = Flask(_name_)
app.secret_key = os.getenv("Flask secret key", "G-API key")  # 

uploaded_files = []
vector_store_created = False

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the context," and don't provide the wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]

@app.route("/", methods=["GET", "POST"])
def home():
    global uploaded_files, vector_store_created
    if request.method == "POST":
        if "pdf_files" not in request.files:
            flash("No files uploaded")
            return redirect(request.url)

        pdf_files = request.files.getlist("pdf_files")
        if not pdf_files:
            flash("No files uploaded")
            return redirect(request.url)

        uploaded_files = pdf_files
        raw_text = get_pdf_text(uploaded_files)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        vector_store_created = True
        flash("PDF processed successfully!")
        return redirect(url_for('home'))

    return render_template("index.html", vector_store_created=vector_store_created)

@app.route("/ask", methods=["POST"])
def ask():
    if request.method == "POST":
        user_question = request.form["question"]
        if not user_question:
            flash("Please enter a question.")
            return redirect(url_for('home'))

        response_text = user_input(user_question)
        return render_template("index.html", response=response_text, vector_store_created=True)

if __name__ == "main":
    app.run(debug=True)