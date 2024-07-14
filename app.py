import streamlit as st
from src.helper import process_in_parallel, get_conversationalchain

def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("User", message.content)
        else:
            st.write("Reply", message.content)

def main():
    st.set_page_config("Langchain_RAG_APP")
    st.header("Langchain Retrieval System")
    user_question = st.text_input("Ask a question from the PDF files")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Upload Your Data")
        pdf_docs = st.file_uploader("Upload your pdf files and click on the Submit and Process Button", accept_multiple_files=True)

        if st.button("Submit and Process"):
            if pdf_docs:
                with st.spinner("Processing...."):
                    vector_stores = process_in_parallel(pdf_docs)
                    st.session_state.conversation = get_conversationalchain(vector_stores[0]) 
                    st.success("Processing Complete")
            else:
                st.warning("Please upload PDF files.")

if __name__ == "__main__":
    main()
