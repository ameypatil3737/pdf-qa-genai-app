import os
import tempfile
import numpy as np
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# 1. Read API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found in .env file")
    st.stop()

client = OpenAI(api_key=api_key)

# 2. Load embedding model once
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# 3. Read text from uploaded PDF
def extract_text_from_pdf(pdf_file_path):
    reader = PdfReader(pdf_file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# 4. Break large text into smaller chunks
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# 5. Convert chunks into embeddings
def create_embeddings(chunks):
    embeddings = embedding_model.encode(chunks)
    return np.array(embeddings).astype("float32")

# 6. Store embeddings in FAISS index
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# 7. Find most relevant chunks for question
def retrieve_relevant_chunks(question, chunks, index, top_k=3):
    question_embedding = embedding_model.encode([question]).astype("float32")
    distances, indices = index.search(question_embedding, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks

# 8. Ask LLM using only retrieved chunks
def ask_llm(question, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant.
Answer the user's question only from the context below.
If the answer is not available in the context, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You answer questions only from provided document context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

# 9. Streamlit user interface
st.set_page_config(page_title="PDF Q&A App", layout="wide")
st.title("PDF Question Answering App")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    with st.spinner("Reading PDF..."):
        document_text = extract_text_from_pdf(tmp_file_path)

    if not document_text.strip():
        st.warning("No readable text found in the PDF.")
        st.stop()

    with st.spinner("Chunking and indexing document..."):
        chunks = chunk_text(document_text)
        embeddings = create_embeddings(chunks)
        index = build_faiss_index(embeddings)

    st.success("PDF processed successfully!")

    question = st.text_input("Ask a question from the PDF:")

    if question:
        with st.spinner("Finding answer..."):
            relevant_chunks = retrieve_relevant_chunks(question, chunks, index, top_k=3)
            answer = ask_llm(question, relevant_chunks)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("See retrieved document chunks"):
            for i, chunk in enumerate(relevant_chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(chunk)