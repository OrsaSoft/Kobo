# from sqlalchemy.orm import sessionmaker
# from sqlalchemy import create_engine, select,not_
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


print(f"LangChain version: {langchain.__version__}") # 0.3.27

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db_path = "./vectordb"

vector_db = Chroma(persist_directory=db_path,embedding_function=embeddings)

if "messages" not in st.session_state:
    st.session_state.messages = []
    mesaj = "You are an assistant for question-answering tasks"
    st.session_state.messages.append(SystemMessage(content=mesaj))


# 6️⃣ Retriever ve LLM kısmı
retriever = vector_db.as_retriever(search_kwargs={"k" : 100})
llm = ChatMistralAI(model_name="magistral-small-2509",api_key="oJ6wgJeUMlciaLyoojF2OUancT1FoOAe")
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen bir yapay zeka asistanısın. Bu Belgeler hakkında sana soru sorulacak {context}"),
    ("human", "{input}"),
])

# Geçmiş mesajları göster
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

prompt = st.chat_input("Your question :")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(content=prompt))






combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rga_chain = create_retrieval_chain(retriever, combine_chain)
print("işlem bitti")

# Sorguyu çalıştır
response = rga_chain.invoke({"input": prompt})

# Süreyi bitir
end = tm.time()


cevap = response["answer"]

with st.chat_message("assistant"):
    st.markdown(cevap)
    st.session_state.messages.append(AIMessage(content=cevap))
    



# What does the US government expect of Bytedance ?


