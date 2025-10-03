def make_question_prompt(question: str) -> str:
    return f"Use all resources at your disposal to answer the following question{question}"


question = make_question_prompt(
    question"Identify traits with lod scores > 3.0 at markers starting with Rsm100000. Tell me what those traits are involved in biology.")
