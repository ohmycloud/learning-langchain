from ragatouille import RAGPretrainedModel
import requests

RAG = RAGPretrainedModel.from_pretrained(
    'colbert-ir/colbertv2.0'
)

def get_wikipedia_page(title: str):
    '''
    Retrieve the full text content of a Wikipedia page.

    :param title: str - Title of the Wikipedia page.
    :return: str - Full text content of the page as raw string.
    '''
    # Wikipedia API endpoint
    URL = 'https://en.wikipedia.org/w/api.php'

    # parameters for the API request
    params = {
        'action': 'query',
        'format': 'json',
        'titles': title,
        'prop': 'extracts',
        'explaintext': True,
    }

    # custom USer-Agent header to comply with Wikipedia's best practices
    headers = {'User-Agent': 'RAGatouille_tutorial/0.0.1'}
    response = requests.get(URL, params=params, headers=headers)
    data = response.json()

    # extracting page content
    page = next(iter(data['query']['pages'].values()))
    return page['extract'] if 'extract' in page else None

full_document = get_wikipedia_page('Hayao_Miyazaki')

# create an index
RAG.index(
    collection=[full_document],
    index_name='Miyazaki-123',
    max_document_length=180,
    split_documents=True,
)

# query
results = RAG.search(query='What animation studio did Miyazaki found?', k=3)

# utiliza langchain retriver
retriver = RAG.as_langchain_retriever(k=3)
res = retriver.invoke('What animation studio did Miyazaki found?')
print(res)
