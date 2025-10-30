# This module formats the query to pass to the system


def make_query_prompt(query: str) -> str:
    return (
        f"Use all resources at your disposal to answer the following question: {query}"
    )

#question = input("We are ready for you. Please ask your question: ")
question = "Identify pairs of traits and markers with lod score > 4. Extract descriptions of traits."

query = make_query_prompt(query=question)
