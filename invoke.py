from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=50,
    base_url="https://api.deepseek.com"
)
system_msg = SystemMessage(
    '''You are a helpful assistant that responds to questions with three
            exclamation marks.'''
)
human_msg = HumanMessage("What is the capital of France?")
message = model.invoke([system_msg, human_msg])
print(message)
