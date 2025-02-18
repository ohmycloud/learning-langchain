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
    return model.invoke(prompt)

llm_output = chatbot.invoke({'question': 'Which model providers offer LLMs?'})
print(llm_output)
