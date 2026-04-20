import os
from operator import index

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader #helps to make us read/load data from different file formats and make them digestable for the llm
from langchain_text_splitters import CharacterTextSplitter #help us with long pieces of text and break them into smaller chnks that the llm can proces more effectively
from langchain_openai import OpenAIEmbeddings #helps us to convert text into numerical vectors that cpture the meaning of the txt, which can then be search and retrieval
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    print("Loading text document")

    loader = TextLoader("C:/Users/rot/Documents/PythonProjects/langchain-course/mediumblog1.txt", encoding='UTF-8')
    document = loader.load()

    print(document)#document variable is a list. that holds langchain document under a single parameter i.e., page_content, we have all the data seperated by ,

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document) # this step is to chunk out mediumblog1.txt into smaller pieces of text, has multiple page_content because of text_spliter config like chunk_size etc
    print(f"created {len(texts)} chunks")
    print(texts)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")

    print("ingesting...")

    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv('INDEX_NAME')) #langchain is going to iterate through all of the chunks, and embedd each and everyone of them and store it in db

    print("ingestion complete")
