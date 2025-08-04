import streamlit as st
import requests

FASTAPI_URL = "http://backend:8000"

st.set_page_config(page_title="Langchain Retrieval System", page_icon="📚")
st.title("📚 Langchain Retrieval System")

# Session state for chat
if "conversation_ready" not in st.session_state:
    st.session_state.conversation_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask a question from the PDF files")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.conversation_ready:
        try:
            response = requests.post(
                f"{FASTAPI_URL}/chat/",
                json={"question": user_input},
                timeout=30
            )
            if response.status_code == 200:
                answer = response.json()["answer"]
            else:
                answer = f"❌ Error: {response.json().get('detail', 'Unknown error')}"
        except Exception as e:
            answer = f"🚫 Failed to connect to backend: {e}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    else:
        with st.chat_message("assistant"):
            st.markdown("⚠️ Please upload and process PDF files from the sidebar first.")

# Sidebar for uploading PDFs
with st.sidebar:
    st.title("Upload Your PDF Data")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if st.button("Submit and Process"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Processing your documents..."):
                try:
                    files = [("files", (file.name, file, "application/pdf")) for file in uploaded_files]
                    response = requests.post(f"{FASTAPI_URL}/process/", files=files, timeout=60)

                    if response.status_code == 200 and response.json().get("status") == "success":
                        st.success("PDFs processed successfully.")
                        st.session_state.conversation_ready = True
                    else:
                        st.error("❌ Error during processing: " + response.json().get("message", "Unknown error"))
                        st.session_state.conversation_ready = False
                except Exception as e:
                    st.error(f"🚫 Failed to connect to backend: {e}")
                    st.session_state.conversation_ready = False
