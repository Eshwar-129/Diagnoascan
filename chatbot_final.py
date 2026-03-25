from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


#OPEN_ROUTER_API="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"
OPEN_ROUTER_API="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"


# =========================
# CREATE CHATBOT
# =========================
def create_chatbot(pdf_paths):

    # -------- Read PDFs --------
    full_text = ""
    for path in pdf_paths:
        reader = PdfReader(path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    # -------- Split --------
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_text(full_text)

    # -------- Embeddings --------
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=OPEN_ROUTER_API,
        base_url="https://openrouter.ai/api/v1"
    )

    db = FAISS.from_texts(chunks, embeddings)

    # -------- LLM --------
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=OPEN_ROUTER_API
    )

    # -------- RAG chain --------
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff"
    )

    return rag_chain, llm, db


# =========================
# ASK QUESTION
# =========================
def ask_question(question, rag_chain, llm, db):

    # Step 1: search documents
    docs = db.similarity_search(question, k=2)

    # Step 2: if relevant docs found → use RAG
    if docs and len(docs[0].page_content.strip()) > 30:
        print("📄 Answering from PDF...")
        result = rag_chain.invoke({"query": question})
        return result["result"]

    # Step 3: fallback → normal chatbot
    else:
        print("🤖 General chatbot answer...")
        return llm.invoke(question).content
