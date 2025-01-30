import os

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import OPENAI_API_KEY

db_path = '.store'

os.makedirs(db_path, exist_ok=True)


embeddings = OpenAIEmbeddings(api_key = OPENAI_API_KEY,model="text-embedding-3-large")



def add_docs(docs):
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(db_path)


def load():
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)