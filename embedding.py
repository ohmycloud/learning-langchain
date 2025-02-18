from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter;
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_postgres import PGVector
from langchain_core.documents import Document
import uuid

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("No API key found. Please check ypur .env file.")

## Load the document
loader = TextLoader("./test.txt")
doc = loader.load()

## Split the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
)
chunks = text_splitter.split_documents(doc)

## Generate embeddings
embeddings_model = OpenAIEmbeddings(
    api_key=api_key,
    base_url='https://vip.apiyi.com/v1'
)
embeddings = embeddings_model.embed_documents(
    [chunk.page_content for chunk in chunks]
)

# embed each chunk and insert it into the vector store
connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
db = PGVector.from_documents(chunks, embeddings_model, connection=connection)

# query
outputs=db.similarity_search("raku", k=4)
for i in outputs:
    print(i)
