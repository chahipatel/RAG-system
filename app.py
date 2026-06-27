from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

chat_history = []

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db_multi",
    embedding_function=embeddings
)

llm = ChatMistralAI(
    model="mistral-small-2506",
    api_key=os.getenv("MISTRAL_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Use chat history and context. If not found, say Not found in document."),
    ("human",
     "Chat History:\n{history}\n\nContext:\n{context}\n\nQuestion:\n{question}")
])

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

def load_pdf(path):
    reader = PyPDFLoader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def load_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return " ".join(soup.get_text(" ").split())

def add_to_db(text, source):
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c, metadata={"source": source}) for c in chunks]
    vectorstore.add_documents(docs)

def ask(question):
    global chat_history

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(question)

    context = "\n\n".join([d.page_content for d in docs])
    history = "\n".join(chat_history[-6:])

    final_prompt = prompt.invoke({
        "history": history,
        "context": context,
        "question": question
    })

    res = llm.invoke(final_prompt)

    chat_history.append(f"User: {question}")
    chat_history.append(f"AI: {res.content}")

    return res.content

def main():
    while True:
        print("\n1. Add PDF\n2. Add URL\n3. Ask\n4. Exit")
        c = input("Enter: ")

        if c == "1":
            p = input("PDF path: ")
            add_to_db(load_pdf(p), "pdf")
            print("Done")

        elif c == "2":
            u = input("URL: ")
            add_to_db(load_url(u), "url")
            print("Done")

        elif c == "3":
            q = input("Question: ")
            print("\n", ask(q))

        elif c == "4":
            break

if __name__ == "__main__":
    main()