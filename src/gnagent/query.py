# This module formats the query to pass to the system


def make_query_prompt(query: str) -> str:
    return (
        f"Use all resources at your disposal to answer the following question: {query}"
    )

#question = input("We are ready for you. Please ask your question: ")
question = "Identify trait variables related to rs51473933 with lod score > 3. Look up their descriptions."

query = make_query_prompt(query=question)
