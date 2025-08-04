from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sys, os
from io import BytesIO

sys.path.append(os.path.dirname(__file__))
from src.helper import process_documents, get_conversationalchain

import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conversation_chain = None

@app.post("/process/")
async def process_pdfs(files: List[UploadFile] = File(...)):
    global conversation_chain

    try:
        logging.info(f"Received {len(files)} file(s) for processing.")
        # Convert UploadFiles (which use async methods) into synchronous file-like objects.
        file_streams = []
        for file in files:
            data = await file.read()  # await the asynchronous read()
            from io import BytesIO
            file_streams.append(BytesIO(data))
        vector_store = process_documents(file_streams)
        conversation_chain = get_conversationalchain(vector_store)
        logging.info("Conversational chain initialized.")
        return {"message": "Documents processed successfully", "status": "success"}
    except Exception as e:
        logging.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

class ChatInput(BaseModel):
    question: str


@app.post("/chat/")
async def chat_with_pdf(data: ChatInput):
    global conversation_chain

    if conversation_chain is None:
        raise HTTPException(status_code=400, detail="Conversation not initialized. Please process documents first.")

    try:
        result = conversation_chain.invoke({"question": data.question})
        return {"answer": result["answer"]}
    except Exception as e:
        logging.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
