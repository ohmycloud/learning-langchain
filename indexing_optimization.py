from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from openai import embeddings
from pydantic import BaseModel
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
import uuid
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("No API key found. Please check ypur .env file.")

connection='postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
collection_name = 'summaries'
embeddings_model = OpenAIEmbeddings(
    api_key=api_key,
    base_url='https://vip.apiyi.com/v1'
)

# load the document
loader = TextLoader('./test.txt', encoding='utf-8')
docs=loader.load()

print('length of loaded docs: ', len(docs[0].page_content))

# split the document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

prompt_text = 'Summarize the following document:\n\n{doc}'
prompt = ChatPromptTemplate.from_template(prompt_text)
llm = ChatOpenAI(
    temperature=0,
    model='gpt-3.5-turbo',
    api_key=api_key,
    base_url='https://vip.apiyi.com/v1'
)

summarize_chain = {
    'doc': lambda x: x.page_content } | prompt | llm | StrOutputParser()

# batch the chain across the chunks
summaries = summarize_chain.batch(
    chunks,
    {'max_concurrency': 5}
)

# the vectorstore to use to index the child chunks
vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

# the storage layer for the parent documents
store = InMemoryStore()
id_key = 'doc_id'

# indexing the summaries in our vector store, whilst
# retaining the original documents in our document store:
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# changed from summaries to chunks since we need same
# length as docs
doc_ids = [str(uuid.uuid4()) for _ in chunks]

# each summary is linked to the original document by the doc_id
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
     for i, s in enumerate(summaries)
]

# add the document summaries to the vector store for similarity search
retriever.vectorstore.add_documents(summary_docs)

# store the original documents in the document store,
# linked to their summaries via doc_ids
# this allows us to first search summaries efficiently,
# then fetch the full docs when needed.
retriever.docstore.mset(list(zip(doc_ids, chunks)))

# vector store retrieves the summaries
sub_docs = retriever.vectorstore.similarity_search(
    'chapter on philosophy', k=2
)

# whereas the retriever will return the larger source document chunks:
retrieved_docs = retriever.invoke('chapter on philosophy')
print(retrieved_docs)
