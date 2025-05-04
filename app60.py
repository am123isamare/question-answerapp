import os
import streamlit as st
import pdfplumber
import pytesseract
import requests
import warnings
from pinecone import Pinecone, ServerlessSpec
from PIL import Image
import re
import docx

warnings.filterwarnings("ignore", category=UserWarning)

# --- 🔐 API Keys
GEMINI_API_KEY = "AIzaSyBtzJylxp0z--wV-XGdAlGmhSIvjlkx7Vg"
PINECONE_API_KEY = "pcsk_2FJh9d_6EsNvxHJdou3swQY3GLRURN3X5TPrJAjJd3znGMfUaAZk8S62cJ1ZKCD7i533fi"
GROQ_API_KEY = "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"

# --- 🧠 Pinecone Setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "gemini-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# --- 📄 Text Extraction

def extract_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    img = page.to_image(resolution=300).original
                    text += pytesseract.image_to_string(img) + "\n"
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

# --- 🔢 Gemini Embeddings

def create_embeddings_with_gemini(text):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text}]}
        }
        response = requests.post(url, headers=headers, json=payload)
        return response.json()["embedding"]["values"]
    except Exception as e:
        print(f"Embedding error: {e}")
        return [0.0] * 768

# --- ⬆️ Index Text

def upsert_to_pinecone(text, file_name):
    embedding = create_embeddings_with_gemini(text)
    metadata = {"file_name": file_name}
    index.upsert(vectors=[(file_name, embedding, metadata)])

# --- 🔍 Query

def query_pinecone(query):
    query_embedding = create_embeddings_with_gemini(query)
    return index.query(vector=query_embedding, top_k=5, include_metadata=True)['matches']

# --- 🧠 Groq Answer

def get_groq_answer(context, question):
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        data = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided text."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Groq error: {e}"

# --- 🚀 Streamlit App

def main():
    st.set_page_config(page_title="📄 AI Q&A", page_icon="🤖")
    st.title("📄📚 File-Based Q&A using Pinecone + Groq")

    uploaded_files = st.file_uploader("Upload PDF, TXT, or DOCX files", accept_multiple_files=True)
    question_input = st.text_input("Ask a question based on uploaded files:")

    doc_texts = {}

    if uploaded_files:
        st.info("📥 Extracting & indexing documents...")
        for file in uploaded_files:
            text = extract_text(file)
            if text.strip():
                doc_texts[file.name] = text
                upsert_to_pinecone(text, file.name)
                st.success(f"✅ {file.name} indexed")
            else:
                st.warning(f"⚠️ No content found in {file.name}")

    if st.button("Get Answer") and question_input:
        results = query_pinecone(question_input)
        results = [r for r in results if r['id'] in doc_texts]

        if results:
            best_match = results[0]
            file_name = best_match['id']
            context_text = doc_texts[file_name]
            answer = get_groq_answer(context_text, question_input)
            st.markdown(f"#### ❓ {question_input}")
            st.success(answer)
            st.caption(f"📁 File: {file_name} | 🔍 Score: {best_match['score']:.4f}")
        else:
            st.warning("⚠️ No relevant content found in the indexed documents.")

if __name__ == "__main__":
    main()