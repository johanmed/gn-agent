"""
This scripts runs through a demo on the use of RAG system for genomic analysis

Embedding model = Qwen/Qwen3-Embedding-0.6B

Generative model = Qwen/Qwen2.5-72B-Instruct-Q4_K_M

Summary model = Phi-3-mini-4k-instruct-fp16

Requirements: langchain, llama-cpp-python (see tapas_env for full details)
"""


# Load document and process

texts =''

with open('document.txt') as file:
    
    read = file.read()
    
    texts = texts + read
    

sentences = texts.split('\n')[:200000]

print('\nDocument ready!')


# Load quantized version of the generative model

from langchain_community.llms import LlamaCpp

gener_model = LlamaCpp(
    model_path = '../Qwen2.5-72B-Instruct-Q4_K_M.gguf',
    max_tokens = 1000,
    n_ctx = 8192, 
    seed = 2025,
    temperature = 0,
    verbose = False)


print('\nGenerative model loaded!')


# Load embedding model from hugging face

from langchain_community.embeddings import HuggingFaceEmbeddings

embed_model = HuggingFaceEmbeddings(
    model_name = 'Qwen/Qwen3-Embedding-0.6B')


print('\nEmbedding model loaded!')


# Set up or load vector database using embedding model

from langchain_community.vectorstores import FAISS

import os

from tqdm import tqdm


db_path = 'vector_db'


def build_load_db(sentences, embed_model, db_path = db_path, chunk_size=500):
	
	if os.path.exists(db_path):
		
		print('\nLoading FAISS vector database from disk...')

		db = FAISS.load_local(
			db_path, 
			embed_model, 
			allow_dangerous_deserialization = True)
		
		return db
	
	else:

		print('\nBuilding FAISS vector database and saving to disk...')
		
		db = None

		for i in tqdm(range(0, len(sentences), chunk_size)):
	
			chunk = sentences[i:i+chunk_size]

			if db is None:
			
				db = FAISS.from_texts(chunk, embed_model) # initialize
	
			else:

				db.add_texts(chunk) # add new chunk
		
		db.save_local(db_path) # save to disk at the end
	
		return db



db = build_load_db(sentences, embed_model)

print('\nAccess to vector database confirmed!')


# Set up RAG prompt

from langchain_core.prompts import PromptTemplate

rag_template = """
<s><|user|>

Relevant information:
{context}

History:
{chat_history}

Provide a concise answer to the question below. Check first in the history above. If you do not find the answer to the question, use the relevant information above. Do not add any external information. 

Think with me step-by-step.

Start by selecting only use cases relevant to the question. Explore different reasoning by emulating a conversation between 3 experts. All experts will write down 1 step of their thinking, then share with the group. Then all experts will go on to the next step and so on. If any expert realizes his wrong at any point, he should leave the conversation.

The question is:
{question}
<|end|>
<|assistant|>
"""

rag_prompt = PromptTemplate(
    input_variables = ['context', 'question', 'summary'],
    template = rag_template)


# Set up retriever prompt

retriever_template = """
<s><|user|>
Given the following conversation, generate a search query to retrieve relevant documents. 

Conversation:
{input}
<|end|>
<|assistant|>
"""


retriever_prompt = PromptTemplate(
	input_variables = ['input'],
	template =retriever_template)
 

# Set up summary prompt for memory propagation

summary_template = """
<s><|user|>

Summarize the conversations and update with the new lines. Be as concise as possible without loosing key information.

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:<|end|>
<|assistant|>
"""

summary_prompt = PromptTemplate(
    input_variables = ['summary', 'new_lines'], 
    template = summary_template)
    
    
    
# Perform conversation summary using a smaller model (Phi-3)

summary_model = LlamaCpp(
    model_path = '../Phi-3-mini-4k-instruct-fp16.gguf',
    max_tokens = 500,
    n_ctx = 2048,
    seed = 2025,
    temperature = 0,
    verbose = False)

from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm = summary_model,
    memory_key = 'chat_history',
    input_key = 'question',
    output_key = 'answer',
    prompt = summary_prompt,
    max_token_limit = 1000,
    return_messages = True)
    
    
    

# Define RAG pipeline


from langchain.chains.history_aware_retriever import create_history_aware_retriever


history_aware_retriever = create_history_aware_retriever(
	retriever = db.as_retriever(),
	llm = gener_model,
	prompt = retriever_prompt)


from langchain.chains.combine_documents import create_stuff_documents_chain

combine_docs_chain = create_stuff_documents_chain(
	llm = gener_model,
	prompt = rag_prompt)


from langchain.chains.retrieval import create_retrieval_chain

retrieval_chain = create_retrieval_chain(
    combine_docs_chain = combine_docs_chain,
    retriever = history_aware_retriever)
    
print('\nPipeline ready!')


# Get generated answer and citations


def rag(question, retrieval_chain = retrieval_chain, memory = memory):
	
	print('\nQuery execution...')
	
	# Get memory content

	memory_var = memory.load_memory_variables({})

	chat_history = memory_var.get('chat_history', '')

	# Execute query

	result = retrieval_chain.invoke(
		{'question': question,
		'input': question, 
		'chat_history': chat_history})

	print('\n Generated_text:\n', result['answer'])

	print('\n Citations:\n', result['context'])

	# Update memory

	memory.save_context(
		{'question': question},
		{'answer': result['answer']})



rag('What is the lod for trait leptin receptor EPFLMouseLiverCDEx0413 at position 100?')
