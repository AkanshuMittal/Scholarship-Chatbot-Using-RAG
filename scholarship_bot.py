import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
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


CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Don't provide anything out of the context

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

# -----------------------------
# Custom Prompt
# -----------------------------
def set_custom_prompt():
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
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

     # RAG chain (only once)
    if "qa" not in st.session_state:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        st.session_state.qa = ConversationalRetrievalChain.from_llm(
            llm=load_llm(),
            retriever=get_vectorstore().as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=False,
            chain_type_kwargs={"prompt": set_custom_prompt()}
        )

    # Display previous messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # User input
    user_input = st.chat_input("Ask me anything about scholarships!")

    if user_input:
        st.chat_message('user').markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:

            # Invoke chain
            response = st.session_state.qa.invoke({'question': user_input})

            answer = response["answer"]
            
            # Display response
            st.chat_message('assistant').markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

