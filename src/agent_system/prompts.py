from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

naturalize_prompt = {
    "messages": [
        SystemMessage(
            """
            You are extremely good at naturalizing RDF and inferring meaning.
            """
        ),
        HumanMessage(
            """
            Take following data and make it sound like Plain English.
            You should return a coherent paragraph with clear sentences.
            Data: "http://genenetwork.org/id/traitBxd_20537:\
            http://purl.org/dc/terms/isReferencedBy: \
            http://genenetwork.org/id/unpublished22893\n \
            http://genenetwork.org/term/locus: \
            http://genenetwork.org/id/Rsm10000002554
            Result:"""
        ),
        AIMessage(
            """
            traitBxd_20537 is referenced by unpublished22893 and has been tested for Rsm10000002554"""
        ),
        HumanMessage(
            """
            Take following RDF data andmake it sound like Plain English.
            You should return a coherent paragraph with clear sentences.
            Data: {text}
            Result:"""
        ),
    ]
}

analyze_prompt = {
    "messages": [
        SystemMessage(
            """
            You are an experienced data analyst who provides accurate and concise feedback. You do extremely well with biological data.
            Answer the question below using provided information.
            Do not modify entities names such as trait and marker.
            Give your response in 200 words max. Do not repeat answers.
            """
        ),
        HumanMessage(
            """
            Context:
            Trait A has a lod score of 2.9 at the marker Rsm1001.
            Trait B has a lod score of 3.5 for Rsm1001.
            History:
            No lod score found for trait A and B so far.
            Question:
            What is the lod score of trait A and B at marker Rsm1001?
            Answer:"""
        ),
        AIMessage(
            """
            Trait A and B have a lod score of 2.9 and 3.5 respectively at\
            Rsm1001."""
        ),
        HumanMessage(
            """
            Context:
            {context}
            History:
            {existing_history}
            Question:
            {input}
            Answer:"""
        ),
    ]
}

check_prompt = {
    "messages": [
        SystemMessage(
            """
            You are an expert in evaluating data relevance in the biological field. You do it seriously.
            Assess if the provided answer can help address the query.
            An answer that addresses a subquestion of the query is still relevant.
            Return strictly "yes" or "no". Do not add anything else."""
        ),
        HumanMessage(
            """
            Answer:
            The lodscore at Rs31201062 for traitBxd_18454 is 4.69
            Query:
            What is the lodscore of traitBxd_18454 at locus Rs31201062?
            Decision:"""
        ),
        AIMessage("yes"),
        HumanMessage(
            """
            Answer:
            {answer}
            Query:
            {input}
            Decision:"""
        ),
    ]
}

summarize_prompt = {
    "messages": [
        SystemMessage(
            """
            You are an excellent and concise summary maker for conversations.
            Summarize in bullet points the conversation below.
            Do not modify entities names such as trait and marker.
            Give your response in 100 words max. Do not repeat answers."""
        ),
        HumanMessage(
            """
            Conversation:
            Trait A has a lod score of 4.7 at marker Rs71192.
            Trait B has a lod score of 1.9 at marker Rs71192.
            Those 2 traits are related to an immune disease.
            Summary:"""
        ),
        AIMessage(
            """
            Two traits involved in diabetes have a lod score of 1.9 and 4.7 at Rs71192."""
        ),
        HumanMessage(
            """
            Conversation:
            {full_context}:
            Summary:"""
        ),
    ]
}

synthesize_prompt = {
    "messages": [
        SystemMessage(
            """
            You are an expert in synthesizing biological information. Compile the following summarized interactions into a single, well-structured paragraph that answers the original question coherently.
            Ensure the response is insightful, concise, and draws logical inferences where possible.
            Provide only the final paragraph, nothing else.
            Give your response in 100 words max. Do not repeat answers."""
        ),
        HumanMessage(
            """
            Original question:
            Take traits and B and compare their lod scores at Rs71192
            Summarized history:
            ["Traits A and B involved in diabetes have a lod score of 1.9 and 4.7 at Rs71192"]
            Conclusion:"""
        ),
        AIMessage(
            """
            The lod score at Rs71192 for trait A (1.9) is less than for trait B (4.7)."""
        ),
        HumanMessage(
            """
            Original question:
            {input}
            Summarized history:
            {updated_history}
            Conclusion:"""
        ),
    ]
}

split_prompt = {
    "messages": [
        SystemMessage(
            """
            You are an expert in genetics. You generate independent subqueries in genetics for parallel processing.
            Based on the context, ask relevant questions that help achieve the task. Make sure the subquestions make fully sense alone. If a marker or trait is shared between the subqueries, make sure to mention it explicitly in each subquery. There should be no words such as "this", "that" or "those" in the subqueries. The goal is to have subquestions that do not have any implicit relationships so that they can be processed separately.
            Return only the subquestions.
            Return strictly a JSON list of strings, nothing else."""
        ),
        HumanMessage(
            """
            Query:
            Identify traits with a lod score > 3.0 for the marker Rsm10000011643. Tell me what this marker is involved in biology.
            Result:"""
        ),
        AIMessage(
            """
            ["What BXDPublish, GWA or GEMMA traits  have a lod score > 3.0 at Rsm10000011643?", "What is Rsm10000011643 involved in biology?"]"""
        ),
        HumanMessage(
            """
            Query:
            Identify traits with a lod score > 3.0 for the marker Rsm10000011643. Find what those traits are.
            Result:"""
        ),
        AIMessage(
            """
            ["What BXDPublish, GWA or GEMMA traits  have a lod score > 3.0 at Rsm10000011643?", "What are BXDPublish, GWA or GEMMA traits involved in biology?"]"""
        ),
        HumanMessage(
            """
            Query:
            {query}
            Result:"""
        ),
    ]
}

finalize_prompt = {
    "messages": [
        SystemMessage(
            """
            You are an experienced geneticist. Given the subqueries and corresponding answers, generate a comprehensive explanation to address the query using all information provided.
            Ensure the response is insightful, concise, and draws logical inferences where possible.
            Do not modify entities names such as trait and marker.            
            Make sure to link based on what is common in the answers.
            Provide only the story, nothing else.
            Do not repeat answers. Use only 200 words max."""
        ),
        HumanMessage(
            """
            Query:
            Identify two traits related to diabetes.
            Compare their lod scores at Rsm149505.
            Subqueries:
            ["Identify two traits related to diabetes",
            "Compare lod scores of same traits at Rsm149505"]
            Answers:
            ["Traits A and B are related to diabetes", \
            "The lod score at Rsm149505 is 2.3 and 3.4 for trait A and B"]
            Conclusion:"""
        ),
        AIMessage(
            """
            Traits A and B are related to diabetes and have a lod score of\
            2.3 and 3.4 at Rsm149505. The two traits could interact via a\
            gene close to the marker Rsm149505."""
        ),
        HumanMessage(
            """
            Query:
            {query}
            Subqueries:
            {subqueries}
            Answers:
            {answers}
            Conclusion:"""
        ),
    ]
}

sup_system_prompt1 = SystemMessage(
    """
            You are a supervisor tasked with managing a conversation between the following workers: [researcher, planner, reflector]. Given the following user request, respond with the worker to act next. Each worker will perform a
task and respond with their results and status. When finished, respond with end."""
)

sup_system_prompt2 = SystemMessage(
    """
            Given the conversation above, who should act next? Or should we end? Select one of: [researcher, planner, reflector, end]"""
)

plan_system_prompt = SystemMessage(
    """
            You are an experienced and powerful task planner for genomic analysis. Generate a list of steps to take to solve the query below. You only have access to a biological researcher but he can dive deep and extract any type of information you need. 
            If the user provides critique, respond with a revised version of your previous attempts."""
)

refl_system_prompt = SystemMessage(
    """
            You are a great science teacher. You have taught for almost 50 years and have a very deep knowledge of biology, genomics and bioinformatics. Generate critique and recommendations for the user's submission.
            Provide detailed recommendations to improve quality, clarity, relevance and satisfaction of peers in your field."""
)
