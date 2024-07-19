import streamlit as st
from src.helper import process_in_parallel, get_conversationalchain

def main():
    st.set_page_config(page_title="Langchain Retrieval System", page_icon=":book:")
    st.title("Langchain Retrieval System")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_question = st.chat_input("Ask a question from the PDF files")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        if st.session_state.conversation:
            response = st.session_state.conversation.invoke({'question': user_question})
            chat_history = response['chat_history']

            # Convert chat history to a list of dictionaries
            formatted_chat_history = []
            for message in chat_history:
                if hasattr(message, 'content'):
                    role = 'user' if message.__class__.__name__ == 'HumanMessage' else 'assistant'
                    formatted_chat_history.append({"role": role, "content": message.content})

            st.session_state.messages = formatted_chat_history

            # Bot's Response
            if formatted_chat_history:
                with st.chat_message("assistant"):
                    st.markdown(formatted_chat_history[-1]['content'])
        else:
            with st.chat_message("assistant"):
                st.markdown("Please upload and process your PDF files first.")

    # Sidebar 
    with st.sidebar:
        st.title("Upload Your Data")
        pdf_docs = st.file_uploader("Upload your PDF files and click on the Submit and Process Button", accept_multiple_files=True)

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
