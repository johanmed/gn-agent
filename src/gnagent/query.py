# This module formats the query to pass to the system


def make_query_prompt(query: str) -> str:
    return (
        f"Use all resources at your disposal to answer the following question: {query}"
    )

#question = input("We are ready for you. Please ask your question: ")
question = "Identify traits that have locus rsm10000000451. Among them, extract those with lod score > 3. Look up their descriptions."

query = make_query_prompt(query=question)
