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

# Store chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Track last uploaded file
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

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
def chunk_text(text, chunk_size=800, overlap=200):
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
def retrieve_relevant_chunks(question, chunks, index, top_k=5):
    question_embedding = embedding_model.encode([question]).astype("float32")
    distances, indices = index.search(question_embedding, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks

# 8. Ask LLM using only retrieved chunks
def ask_llm(question, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful document assistant.

Answer the user's question only from the context below.

Rules:
1. Do not make up anything.
2. If the answer is not present in the context, say:
   "I could not find the answer in the document."
3. Answer in simple and clear language.
4. If possible, give the answer in short bullet points.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You answer questions only from provided document context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content

# 9. Streamlit user interface
st.set_page_config(page_title="PDF Q&A App", layout="wide")
st.title("Document Question Answering Assistant")

# Sometimes the user wants to:
# upload same PDF and start fresh
# ask new set of questions
# clean old chat
# This button does that.
if st.button("Clear Chat History"):
    st.session_state.chat_history = []  # this clears old chat
    st.rerun()

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:

    # If new PDF uploaded -> clear chat history
    if st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.chat_history = []  # this clears old chat
        st.session_state.last_uploaded_file = uploaded_file.name

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Read PDF
    with st.spinner("Reading PDF..."):
        document_text = extract_text_from_pdf(tmp_file_path)

    if not document_text.strip():
        st.warning("No readable text found in the PDF.")
        st.stop()

    # Chunking + indexing document
    with st.spinner("Chunking and indexing document..."):
        chunks = chunk_text(document_text)
        embeddings = create_embeddings(chunks)
        index = build_faiss_index(embeddings)

st.success("PDF processed successfully!")

with st.form("question_form", clear_on_submit=True):
    question = st.text_input("Ask a question from the PDF:")
    submit_button = st.form_submit_button("Ask")

if submit_button and question and question.strip():
    with st.spinner("Finding answer..."):
        relevant_chunks = retrieve_relevant_chunks(question, chunks, index, top_k=5)
        answer = ask_llm(question, relevant_chunks)

    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "sources": relevant_chunks
    })
# Display full chat history
if st.session_state.chat_history:
    st.subheader("Chat History")

    for i, chat in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"### Question {i}")
        st.write(chat["question"])

        st.markdown("**Answer:**")
        st.write(chat["answer"])

        st.markdown("**Sources Used:**")
        for j, chunk in enumerate(chat["sources"], 1):
            st.markdown(f"**Source {j}:** {chunk[:300]}...")

        with st.expander(f"See full retrieved chunks for Question {i}"):
            for j, chunk in enumerate(chat["sources"], 1):
                st.markdown(f"**Chunk {j}:**")
                st.write(chunk)