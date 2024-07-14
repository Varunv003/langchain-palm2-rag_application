import os
import time
import logging
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import concurrent.futures
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY



def get_pdf_text(pdf_doc):
    start_time = time.time()
    logging.info("Starting PDF text extraction")
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    logging.info(f"Completed PDF text extraction in {time.time() - start_time} seconds")
    return text


def get_text_chunks(text):
    start_time = time.time()
    logging.info("Starting text chunking")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    logging.info(f"Completed text chunking in {time.time() - start_time} seconds")
    return chunks


def get_vector_store(text_chunks):
    start_time = time.time()
    logging.info("Starting FAISS vector store creation")
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    logging.info(f"Completed FAISS vector store creation in {time.time() - start_time} seconds")
    return vector_store

def get_conversationalchain(vector_store):
    start_time = time.time()
    logging.info("Starting conversational chain setup")
    llm_model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(llm=llm_model, retriever=vector_store.as_retriever(), memory=memory)
    logging.info(f"Completed conversational chain setup in {time.time() - start_time} seconds")
    return conversational_chain

def process_in_parallel(pdf_docs):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        text_futures = [executor.submit(get_pdf_text, pdf_doc) for pdf_doc in pdf_docs]
        texts = [future.result() for future in text_futures]

        chunks_futures = [executor.submit(get_text_chunks, text) for text in texts]
        chunks = [future.result() for future in chunks_futures]

        vector_store_futures = [executor.submit(get_vector_store, chunk) for chunk in chunks]
        vector_stores = [future.result() for future in vector_store_futures]

    return vector_stores