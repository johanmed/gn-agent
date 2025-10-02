retriever_template = f"""
You are powerful data retriever and you strictly return what is asked for.

Given the following conversation, generate a search query to retrieve relevant documents.
Conversation:
{input}
"""

rag_template = f"""
<|im_start|>system
You are a very skilled analyst that can answer concisely questions based on information you have.
<|im_end|>
<|im_start|>user
Answer the question below using following information.
Try using the history first.
If you cannot provide answer with it, then use the context
Context: {context}
History: {chat_history}
Question: {question}
Answer:
<|im_end|>
<|im_start|>assistant
"""

summary_template = f"""
<|system|>
You are a powerful summary assistant. You do as much as you can and if cant take all information, you prioritize by order of importance.
<|end|>
<|user|>
Summarize the conversations and update with the new lines. Be as concise as possible without loosing key information.
Current summary:
{summary}
New lines of conversation:
{new_lines}
New summary:
<|end|>
<|assistant|>
"""
