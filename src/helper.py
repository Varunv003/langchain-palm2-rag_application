import os
import time
import logging
import fitz  # PyMuPDF
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load the .env file from the project root explicitly
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY is not set. Please add it to your .env file.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def get_pdf_text(pdf_doc):
    start_time = time.time()
    pdf_name = pdf_doc.name if hasattr(pdf_doc, 'name') else 'Uploaded File'
    logging.info(f"Starting PDF text extraction for {pdf_name}")
    text = ""

    try:
        if hasattr(pdf_doc, "seek"):
            pdf_doc.seek(0)
            doc = fitz.open(stream=pdf_doc.read(), filetype="pdf")
        else:
            doc = fitz.open(pdf_doc)

        logging.info(f"PDF has {len(doc)} pages")
        for i, page in enumerate(doc):
            page_text = page.get_text()
            if page_text:
                text += page_text
            else:
                logging.warning(f"No text extracted for page {i} of {pdf_name}")
    except Exception as e:
        logging.error(f"Error reading {pdf_name}: {e}")

    logging.info(f"Completed PDF text extraction in {time.time() - start_time:.2f} seconds for {pdf_name}. Length extracted: {len(text)}")
    logging.debug(f"Extracted text snippet: {text[:100]}")
    return text

def get_text_chunks(text):
    start_time = time.time()
    logging.info("Starting text chunking")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    logging.info(f"Completed text chunking in {time.time() - start_time:.2f} seconds, produced {len(chunks)} chunks")
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        logging.error("No text chunks provided for vector store creation!")
        raise ValueError("Empty list of text chunks.")

    start_time = time.time()
    logging.info("Starting FAISS vector store creation")

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001", api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    logging.info(f"Completed FAISS vector store creation in {time.time() - start_time:.2f} seconds")
    return vector_store

def get_conversationalchain(vector_store):
    start_time = time.time()
    logging.info("Starting conversational chain setup")

    llm_model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    logging.info(f"Completed conversational chain setup in {time.time() - start_time:.2f} seconds")
    return conversational_chain

def process_documents(pdf_docs):
    all_chunks = []

    for pdf_doc in pdf_docs:
        text = get_pdf_text(pdf_doc)
        logging.info(f"Extracted text length: {len(text)}")
        chunks = get_text_chunks(text)
        valid_chunks = [chunk for chunk in chunks if chunk.strip()]
        logging.info(f"Valid chunks count for this PDF: {len(valid_chunks)}")
        all_chunks.extend(valid_chunks)

    if not all_chunks:
        logging.error("No valid text chunks found for vector store creation!")
        raise ValueError("Empty list of valid text chunks after filtering.")

    vector_store = get_vector_store(all_chunks)
    return vector_store
