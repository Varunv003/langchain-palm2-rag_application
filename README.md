# DocChat: Langchain Retrieval System

This Streamlit application implements a Langchain-based retrieval system for processing PDF documents and conducting conversational retrieval using Langchain's capabilities.

![LangChain Retrieval Generation](readme_img_1.png)
![RAG Streamlit APP](readme_img_2.png)

## Overview

The application allows users to upload PDF files, extract text, split it into chunks, generate embeddings using Google Palm embeddings, and create a conversational retrieval chain. Users can then ask questions related to the processed PDF content and receive responses based on the conversational chain set up.

### Key Technologies Used

- **Langchain**: A library for natural language processing tasks, including text splitting and conversational retrieval.
- **Google Palm Embeddings**: Embeddings used for semantic similarity and text representation.
- **FAISS (Facebook AI Similarity Search)**: An efficient library for similarity search and clustering of dense vectors.

## Project Setup

### Prerequisites

1. **Python Environment**: Make sure you have Python 3.x installed.
   
2. **Environment Variables**: Create a `.env` file in the project root directory with the following content:
     GOOGLE_API_KEY=your_google_api_key_here
     Replace `your_google_api_key_here` with your actual Google API key.

### Installation

1. **Clone the Repository**: Clone this repository to your local machine:
```bash
git clone https://github.com/Varunv003/langchain-palm2-rag_application
```
2. **Set Up Virtual Environment**: It's recommended to use a virtual environment to manage dependencies:

```bash

python -m venv venv
# On Windows: .\venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate
```
3. **Install Dependencies**: Install required Python packages using pip:

```bash

pip install -r requirements.txt
```
4. **Template Structure**: To set up the initial folder structure of the project, run:

```bash

python template.py
# This command will create necessary directories and files based on your project needs.
```

5. **Running the Application**
To run the Streamlit application:

```bash

streamlit run app.py
# The application will start, and you can access it in your web browser at http://localhost:8501.
```

### File Structure
- **app.py**: Main Streamlit application code for uploading PDFs, processing them, and managing user interactions.
- **helper.py**: Contains helper functions for PDF text extraction, text chunking, FAISS vector store creation, and conversational chain setup.
- **template.py**: Script to initialize the folder structure and create necessary directories/files for the project.
.env: Environment variable file for storing sensitive data like API keys.
### Usage
- Upload PDF Files: Use the "Upload Your Data" sidebar to upload one or more PDF files.
- Process PDFs: Click "Submit and Process" to extract text, generate embeddings, and set up a conversational retrieval chain.
- Ask Questions: Enter questions related to the uploaded PDF content in the text input field.
- View Responses: Responses generated by the Langchain conversational model will be displayed in the main interface.
- Logging: Logging is implemented to capture key steps and timings during PDF text extraction, text chunking, vector store creation, and conversational chain setup. Logs are displayed in the console or terminal where the application is run.

### Future Improvements
- Enhance error handling and user feedback during file upload and processing.
- Improve scalability and performance optimizations for handling larger PDF documents.
- Integrate additional AI models or refine existing models for better conversational responses.