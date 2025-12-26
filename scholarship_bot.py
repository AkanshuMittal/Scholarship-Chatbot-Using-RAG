import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

load_dotenv()


# FAISS vectorstore path
DB_FAISS_PATH = "vectorstore/db_faiss"


# Load OpenAI API key from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# Streamlit cache to load vectorstore once

@st.cache_resource
def get_vectorstore():
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# -----------------------------
# Custom Prompt
# -----------------------------
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

# -----------------------------
# Load OpenAI Chat LLM
# -----------------------------
def load_llm(model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=512):
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return llm

# -----------------------------
# Streamlit Chatbot UI
# -----------------------------
def main():
    st.title("Scholarship Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # User input
    prompt = st.chat_input("Ask me anything about scholarships!")

    if prompt:
        st.chat_message('user').markdown(prompt)

        # Custom prompt template
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Don't provide anything out of the context

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        try:
            # Load FAISS vectorstore
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(model_name="gpt-3.5-turbo"),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Invoke chain
            response = qa_chain.invoke({'query': prompt})

            # Extract results
            result = response["result"]
            source_documents = response["source_documents"]
            result_to_show = result + "\n\nSource Docs:\n" + str(source_documents)

            # Display response
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

