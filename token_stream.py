from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant.'),
        ('human', '{question}'),
    ]
)

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    base_url="https://api.deepseek.com"
)

@chain
def chatbot(values):
    prompt = template.invoke(values)
    for token in model.stream(prompt):
        yield token

llm_output = chatbot.stream({'question': 'Which model providers offer LLMs?'})
for part in llm_output:
    print(part)
