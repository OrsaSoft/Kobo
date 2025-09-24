
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

# Embeddings ve vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_path = "vectordb"
vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
retriever = vector_db.as_retriever()

# Streamlit session_state ile mesaj geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []  # Burada sadece (soru, cevap) tuple listesi
    st.session_state.messages_history = []  # Kullanıcı ve asistan mesajları için

# Önceki mesajları göster
for human_msg, ai_msg in st.session_state.messages_history:
    with st.chat_message("user"):
        st.markdown(human_msg)
    with st.chat_message("assistant"):
        st.markdown(ai_msg)

# Kullanıcıdan input al
prompt = st.chat_input("Your question:")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    # LLM, sistem prompt ile başlatılıyor
    llm = ChatMistralAI(
        model_name="magistral-small-2509",
        api_key=os.environ.get("oJ6wgJeUMlciaLyoojF2OUancT1FoOAe"),
        system_prompt="You are an assistant for question-answering tasks"
    )

    # QA chain oluştur
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Chain'i çalıştır, chat_history sadece (soru, cevap) tuple listesi
    result = qa_chain({
        "question": prompt,
        "chat_history": st.session_state.messages
    })

    answer = result["answer"]

    with st.chat_message("assistant"):
        st.markdown(answer)

    # Mesajları güncelle
    st.session_state.messages.append((prompt, answer))
    st.session_state.messages_history.append((prompt, answer))








