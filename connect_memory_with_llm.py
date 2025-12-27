import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

# Load OpenAI API Key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# Setup LLM (OpenAI Chat Model)
def load_llm(model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=512):
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return llm


# Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt


# Load FAISS Vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Run Query
user_query = input("Write YOUR QUERY HERE: ")
response = qa_chain.invoke({"query": user_query})

print("\nRESULT: ", response["result"])
print("\nSOURCE DOCUMENTS: ", response["source_documents"])




