def make_query_prompt(question: str) -> str:
    return f"Use all resources at your disposal to answer the following question: {question}"


query = make_query_prompt(
    question="Find all QTLs tagged 'new QTL' on chromosome 1. Given that QTL have long names starting with GEMMAMapped_LOCO_BXDPublish, extract names of matching QTLs. Identify markers they are associated to with a lod score > 4."
)
