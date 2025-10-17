def make_query_prompt(question: str) -> str:
    return f"Use all resources at your disposal to answer the following question: {question}"


query = make_query_prompt(
    question="Find 10 QTLs tagged 'new QTL'. Given that QTL have long names starting with GEMMAMapped_LOCO_BXDPublish, extract names of matching QTLs."
)
