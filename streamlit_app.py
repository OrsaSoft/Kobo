
import time as tm
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.agent_toolkits import create_retriever_tool
import langchain
from langchain_community.embeddings import OllamaEmbeddings
from hashlib import sha256
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
import streamlit as st 
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain.chains import ConversationalRetrievalChain

print(f"LangChain version: {langchain.__version__}") # 0.3.27

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

api_key = os.environ.get("oJ6wgJeUMlciaLyoojF2OUancT1FoOAe")

db_path = "vectordb"

vector_db = Chroma(persist_directory=db_path,embedding_function=embeddings)

retriever = vector_db.as_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []
    mesaj = "You are an assistant for question-answering tasks"
    st.session_state.messages.append(SystemMessage(content=mesaj))

for message in st.session_state.messages:
    if isinstance(message,HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    
    elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

prompt = st.chat_input("Your question : ")

if prompt:
    with st.chat_message("user"):
          st.markdown(prompt)
          st.session_state.messages.append(HumanMessage(content=prompt))

    llm = ChatMistralAI(model_name="magistral-small-2509",api_key="oJ6wgJeUMlciaLyoojF2OUancT1FoOAe")
     
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever,return_source_documents=True)

    result = qa_chain({"question": prompt,"chat_history": st.session_state.messages})

    with st.chat_message("assistant"):
         st.markdown(result["answer"])
         st.session_state.messages.append(AIMessage(content=result["answer"]))
         







