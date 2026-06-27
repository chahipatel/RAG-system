import streamlit as st
from app import load_pdf, load_url, add_to_db, ask
import tempfile
import os

st.set_page_config(
    page_title="RAG System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG System")
st.markdown("Ask questions from **PDFs** and **Web URLs** using **Mistral AI + ChromaDB + LangChain**")

# Sidebar
st.sidebar.title("📚 Knowledge Source")

source = st.sidebar.radio(
    "Choose Source",
    ["Upload PDF", "Website URL"]
)

# ---------------- PDF ---------------- #

if source == "Upload PDF":

    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.sidebar.button("Add PDF"):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:

                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            with st.spinner("Reading PDF..."):

                text = load_pdf(pdf_path)
                add_to_db(text, uploaded_file.name)

            os.remove(pdf_path)

            st.sidebar.success("✅ PDF added successfully!")

# ---------------- URL ---------------- #

else:

    url = st.sidebar.text_input("Website URL")

    if st.sidebar.button("Add URL"):

        with st.spinner("Loading Website..."):

            text = load_url(url)
            add_to_db(text, url)

        st.sidebar.success("✅ Website added successfully!")

# ---------------- Chat ---------------- #

st.divider()

st.header("💬 Ask Questions")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask anything from your uploaded knowledge base...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = ask(question)

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )