import os
import tempfile
import hashlib
import numpy as np
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

st.set_page_config(page_title="Financial Document AI Assistant", layout="wide")
# -----------------------------
# 1. Load API key
# -----------------------------
import os
import streamlit as st
from openai import OpenAI

# Try Streamlit secrets first, fallback to env (for local)
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found. Please add it in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# -----------------------------
# 2. Session state
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_uploaded_files_signature" not in st.session_state:
    st.session_state.last_uploaded_files_signature = None

if "document_summary" not in st.session_state:
    st.session_state.document_summary = ""

# -----------------------------
# 3. Load embedding model once
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# -----------------------------
# 4. Utility: make file signature
# -----------------------------
def get_files_signature(uploaded_files):
    signature_text = "||".join([f"{file.name}_{file.size}" for file in uploaded_files])
    return hashlib.md5(signature_text.encode()).hexdigest()

# -----------------------------
# 5. Extract text from PDF
# -----------------------------
def extract_text_from_pdf(pdf_file_path, file_name):
    reader = PdfReader(pdf_file_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            pages.append(
                {
                    "file_name": file_name,
                    "page_number": page_number,
                    "text": page_text.strip(),
                }
            )

    return pages

# -----------------------------
# 6. Chunk text
# -----------------------------
def chunk_text(pages, chunk_size=1000, overlap=100):
    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]
        file_name = page["file_name"]

        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(
                    {
                        "file_name": file_name,
                        "page_number": page_number,
                        "text": chunk,
                    }
                )

            start += chunk_size - overlap

    return chunks

# -----------------------------
# 7. Create embeddings
# -----------------------------
def create_embeddings(chunks):
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(chunk_texts)
    return np.array(embeddings).astype("float32")

# -----------------------------
# 8. Build FAISS index
# -----------------------------
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# -----------------------------
# 9. Cache chunks + embeddings + index
# -----------------------------
@st.cache_resource
def process_documents_cached(file_data):
    all_pages = []

    for file_name, file_bytes in file_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name

        pages = extract_text_from_pdf(tmp_file_path, file_name)
        all_pages.extend(pages)

        try:
            os.remove(tmp_file_path)
        except Exception:
            pass

    chunks = chunk_text(all_pages)
    embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)

    return all_pages, chunks, index

# -----------------------------
# 10. Retrieve relevant chunks
# -----------------------------
def retrieve_relevant_chunks(question, chunks, index, top_k=8, final_k=5):
    question_embedding = embedding_model.encode([question]).astype("float32")
    distances, indices = index.search(question_embedding, top_k)

    retrieved_chunks = []
    seen_texts = set()

    for idx in indices[0]:
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx]
        cleaned_text = " ".join(chunk["text"].split())

        if cleaned_text not in seen_texts:
            seen_texts.add(cleaned_text)
            retrieved_chunks.append(chunk)

        if len(retrieved_chunks) >= final_k:
            break

    return retrieved_chunks

# -----------------------------
# 11. Ask LLM
# -----------------------------
def ask_llm(question, context_chunks):
    not_found_message = (
        "I could not find the answer in the uploaded documents. "
        "Please try rephrasing your question or ask something more specific."
    )

    if not context_chunks:
        return not_found_message

    context = "\n\n".join(
        [
            f"(File: {chunk['file_name']} | Page {chunk['page_number']}) {chunk['text']}"
            for chunk in context_chunks
        ]
    )

    source_items = []
    seen_sources = set()

    for chunk in context_chunks:
        source_key = (chunk["file_name"], chunk["page_number"])
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            source_items.append(f"{chunk['file_name']} - Page {chunk['page_number']}")

    sources_str = "; ".join(source_items)

    prompt = f"""
You are a professional multi-document assistant.

Answer the user's question using only the provided document context.

Rules:
1. Use only the uploaded document context.
2. Do not make up or assume anything.
3. If the answer is not present, say exactly:
   "{not_found_message}"
4. Keep the answer clear, concise, and professional.
5. Use short bullet points where helpful.
6. At the end, always add:
   Source: {sources_str}

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
                "content": "You answer questions only from provided uploaded document context.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content.strip()

    if "Source:" not in answer:
        answer += f"\n\nSource: {sources_str}"

    return answer

# -----------------------------
# 12. Summarize document
# -----------------------------
def summarize_document(chunks, max_chunks=10):
    if not chunks:
        return "No readable content found in the uploaded documents."

    selected_chunks = chunks[:max_chunks]
    context = "\n\n".join(
        [
            f"(File: {chunk['file_name']} | Page {chunk['page_number']}) {chunk['text']}"
            for chunk in selected_chunks
        ]
    )

    prompt = f"""
You are a professional document assistant.

Summarize the uploaded documents using only the content below.

Rules:
1. Do not make up anything.
2. Keep the summary simple and professional.
3. Use short bullet points.
4. Mention the main topics covered.
5. If multiple files are uploaded, mention file names where relevant.

Document Content:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You summarize documents only from provided content.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()

# -----------------------------
# 13. Streamlit UI
# -----------------------------
#st.set_page_config(page_title="Financial Document AI Assistant", layout="wide")
st.image("header.png")
#st.title("Financial Document AI Assistant")
st.markdown(
    """
    Upload one or more PDF documents and ask questions grounded in their content.

    **Note:** Answers are generated only from uploaded documents.
    """
)

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.document_summary = ""
        st.rerun()


with st.sidebar:
    st.header("About this app")
    st.write(
        "This is a multi-PDF RAG application built using Streamlit, FAISS, "
        "Sentence Transformers, and OpenAI."
    )
    st.markdown("---")
    st.write("**Built by:** Amey Anant Patil")
    st.write("GenAI Enthusiast")

if uploaded_files:
    current_signature = get_files_signature(uploaded_files)

    if st.session_state.last_uploaded_files_signature != current_signature:
        st.session_state.chat_history = []
        st.session_state.document_summary = ""
        st.session_state.last_uploaded_files_signature = current_signature

    file_data = [(file.name, file.getvalue()) for file in uploaded_files]

    with st.spinner("Processing uploaded documents..."):
        document_pages, chunks, index = process_documents_cached(tuple(file_data))

    with st.sidebar:

        st.subheader("Document Stats")
        st.write(f"Files uploaded: {len(uploaded_files)}")
        st.write(f"Total pages: {len(document_pages)}")
        st.write(f"Total chunks: {len(chunks)}")


    if not document_pages:
        st.warning("No readable text was found in the uploaded PDF documents.")
        st.stop()

    st.subheader("Uploaded Documents")
    for file in uploaded_files:
        st.write(f"• {file.name}")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Summarize Documents"):
            with st.spinner("Generating summary..."):
                st.session_state.document_summary = summarize_document(chunks)

    if st.session_state.document_summary:
        st.subheader("Document Summary")
        st.markdown(st.session_state.document_summary)

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("Ask a question from the uploaded documents")
        submit_button = st.form_submit_button("Ask")

    if submit_button and question and question.strip():
        with st.spinner("Generating answer..."):
            relevant_chunks = retrieve_relevant_chunks(
                question=question,
                chunks=chunks,
                index=index,
                top_k=8,
                final_k=5,
            )
            answer = ask_llm(question, relevant_chunks)

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer,
                "sources": relevant_chunks,
            }
        )
# -----------------------------
# 14. Display chat history
# -----------------------------
if st.session_state.chat_history:
    st.subheader("Conversation")

    for i, chat in enumerate(st.session_state.chat_history, start=1):
        with st.container():
            st.markdown(f"**Question {i}**")
            st.write(chat["question"])

            st.markdown("**Answer**")
            st.markdown(chat["answer"])

            unique_source_labels = []
            seen_sources = set()

            for chunk in chat["sources"]:
                source_key = (chunk["file_name"], chunk["page_number"])
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    unique_source_labels.append(
                        f"{chunk['file_name']} - Page {chunk['page_number']}"
                    )

            if unique_source_labels:
                st.markdown("**Retrieved Source Pages**")
                for source_label in unique_source_labels:
                    st.info(source_label)

            st.markdown("---")