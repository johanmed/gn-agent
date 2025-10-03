def make_question(task: str, context: str) -> str:
    return f"""
    Task: {task}
    Context: {context}
    """


question = make_question(
    task="Identify traits with lod scores > 3.0 at markers starting with Rsm100000. Tell me what those traits are involved in biology.",
    context="Traits and markers are different. A trait is related to the BXDPublish dataset while a marker is not. The goal is to extract what we know in biology about the traits previously mentioned and link them based on the markers.",
)
