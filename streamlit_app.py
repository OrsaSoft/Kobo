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



# ----------------------
# Başlık ve açıklama
# ----------------------
st.title("📚 AI Chatbot with LangChain + Mistral")
st.write("Belgeler üzerinde sorular sorabilirsiniz. Çıkmak için 'çık' yazın.")

# ----------------------
# LangChain ve Vektör DB ayarları
# ----------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_path = "./vectordb"
vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

retriever = vector_db.as_retriever(search_kwargs={"k": 100})

llm = ChatMistralAI(
    model_name="magistral-small-2509",
    api_key="oJ6wgJeUMlciaLyoojF2OUancT1FoOAe"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen bir yapay zeka asistanısın. Bu belgeler hakkında sorular sorulacak: {context}"),
    ("human", "{input}"),
])

combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rga_chain = create_retrieval_chain(retriever, combine_chain)

# ----------------------
# Kullanıcı girişi ve chatbot yanıtı
# ----------------------
user_input = st.text_input("Soru yazın:")

if user_input:
    if user_input.lower() in ["1", "çık", "exit"]:
        st.stop()
    else:
        # Belgelerden ilgili içerikleri çek
        docs = retriever.get_relevant_documents(user_input)
        st.write(f"Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs, 1):
            st.write(f"--- Document {i} ---")
            st.write(doc.page_content[:500])  # ilk 500 karakter
            st.write("Metadata:", doc.metadata)

        # LLM ile cevap üret
        response = rga_chain.invoke({"input": user_input})
        st.write("🤖 AI'nın cevabı:")
        st.write(response["answer"])


