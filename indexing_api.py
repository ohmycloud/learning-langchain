from langchain.indexes import SQLRecordManager, index
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("No API key found. Please check ypur .env file.")

connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
collection_name = 'raku_docs'
embedding_model = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key,
    base_url='https://vip.apiyi.com/v1'
)
namespace = 'raku_docs_namespace'
vectorstore = PGVector(
    embeddings=embedding_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

record_manager = SQLRecordManager(
    namespace,
    db_url=connection,
)

# create the schema if it doesn't exist
record_manager.create_schema()

# create documents
docs = [
    Document(page_content='there are cats in the pond', metadata={'id': 1, 'source': 'cats.txt'}),
    Document(page_content='ducks are also found in the pond', metadata={'id': 2, 'source': 'ducks.txt'}),
]

# Index the documents
index_1 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup='incremental',  # prevent duplicate documents
    source_id_key='source', # use the source field as the source_id
)

print('Index attempt 1:', index_1)

# second time you attempt to index, it will not add the documents again
index_2 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup='incremental',
    source_id_key='source',
)
print('Index attempt 2:', index_2)

# If we mutate a document, the new version will be written and all old versions
# sharing the same source will be deleted.
docs[0].page_content = 'I just modified this document!'

index_3 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup='incremental',
    source_id_key='source',
)

print('Index attempt 3:', index_3)

# output:
# Index attempt 1: {'num_added': 2, 'num_updated': 0, 'num_skipped': 0, 'num_deleted': 0}
# Index attempt 2: {'num_added': 0, 'num_updated': 0, 'num_skipped': 2, 'num_deleted': 0}
# Index attempt 3: {'num_added': 1, 'num_updated': 0, 'num_skipped': 1, 'num_deleted': 1}
