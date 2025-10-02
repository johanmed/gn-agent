# gn-rag
gn-rag is a rag system designed for genenetwork. 
Work still in progress.

## Description
gn-rag aims at improving search and large-scale data analysis in genenetwork using LLMs capabilities. 
The system uses:
1. A large generative model for question-answering
2. An embedding model and BM25 algorithm for document search and retrieval
3. A small generative model for conversation summarization

## Set up
You will need a Python environment with a number of dependencies to run this project. Tools required include:
- langchain
- langgraph
- llama-cpp-python
- chroma-db
- sparqlwrapper

You dont need to install them one by one. Create your own Python environment and do `pip install -r requirements.txt`
