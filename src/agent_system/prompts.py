naturalize_prompt = f"""
            <|im_start|>system
            You are extremely good at naturalizing RDF and inferring meaning.
            <|im_end|>
            <|im_start|>user
            Take following data and make it sound like Plain English.
            You should return a coherent paragraph with clear sentences.
            Data: "http://genenetwork.org/id/traitBxd_20537:\
            http://purl.org/dc/terms/isReferencedBy: \
            http://genenetwork.org/id/unpublished22893\n \
            http://genenetwork.org/term/locus: \
            http://genenetwork.org/id/Rsm10000002554"
            <|im_end|>
            <|im_start|>assistant
            Result: "traitBxd_20537 is referenced by unpublished22893 and has been tested for Rsm10000002554"
            <|im_end|>
            <|im_start|>user
            Take following RDF data andmake it sound like Plain English.
            You should return a coherent paragraph with clear sentences.
            Data: {text}
            <|im_start|>end
            <|im_start|>assistant"""

analyze_prompt = f"""
             <|im_start|>system
             You are an experienced data analyst who provides accurate and concise feedback. You do extremely well with biological data.
             Answer the question below using provided information.
             Do not modify entities names such as trait and marker.
             Give your response in 200 words max. Do not repeat answers.
             <|im_end|>
             <|im_start|>user
             Context:
             Trait A has a lod score of 2.9 at the marker Rsm1001.
             Trait B has a lod score of 3.5 for Rsm1001.
             History:
             No lod score found for trait A and B so far.
             Question:
             What is the lod score of trait A and B at marker Rsm1001?
             Answer:
             <|im_end|>
             <|im_start|>assistant
             Trait A and B have a lod score of 2.9 and 3.5 respectively at\
             Rsm1001.
             <|im_end|>
             <|im_start|>user
             Context:
             {context}
             History:
             {existing_history}
             Question:
             {state["input"]}
             Answer:
             <|im_end|>
             <|im_start|>assistant"""

check_prompt = f"""
            <|system|>
            You are an expert in evaluating data relevance in the biological field. You do it seriously.
            Assess if the provided answer can help address the query.
            An answer that addresses a subquestion of the query is still relevant.
            Return strictly "yes" or "no". Do not add anything else.
            <|end|>
            <|user|>
            Answer:
            The lodscore at Rs31201062 for traitBxd_18454 is 4.69
            Query:
            What is the lodscore of traitBxd_18454 at locus Rs31201062?
            Decision:
            <|end|>
            <|assistant|>
            yes
            <|end|>
            <|user|>
            Answer:
            {answer}
            Query:
            {state["input"]}
            Decision:
            <|end|>
            <|assistant|>"""

summarize_prompt = f"""
            <|system|>
            You are an excellent and concise summary maker for conversations.
            Summarize in bullet points the conversation below.
            Do not modify entities names such as trait and marker.
            Give your response in 100 words max. Do not repeat answers.
            <|end|>
            <|user|>
            Conversation:
            Trait A has a lod score of 4.7 at marker Rs71192.
            Trait B has a lod score of 1.9 at marker Rs71192.
            Those 2 traits are related to an immune disease.
            Summary:
            <|end|>
            <|assistant|>
            Two traits involved in diabetes have a lod score of 1.9 and 4.7 at Rs71192.
            <|end|>
            <|user|>
            Conversation:
            {full_context}:
            Summary:
            <|end|>
            <|assistant|>"""

synthesize_prompt = f"""
            <|im_start|>system
            You are an expert in synthesizing biological information. Compile the following summarized interactions into a single, well-structured paragraph that answers the original question coherently.
            Ensure the response is insightful, concise, and draws logical inferences where possible.
            Provide only the final paragraph, nothing else.
            Give your response in 100 words max. Do not repeat answers.
            <|im_end|>
            <|im_start|>user
            Original question:
            Take traits and B and compare their lod scores at Rs71192
            Summarized history:
            ["Traits A and B involved in diabetes have a lod score of 1.9 and 4.7 at Rs71192"]
            Conclusion:
            <|im_end|>
            <|im_start|>assistant
            The lod score at Rs71192 for trait A (1.9) is less than for trait B (4.7).
            <|im_end|>
            <|im_start|>user
            Original question:
            {state["input"]}
            Summarized history:
            {updated_history}
            Conclusion:
            <|im_end|>
            <|im_start|>assistant"""

split_prompt = f"""
            <|im_start|>system
            You are an expert in genetics. You generate independent subqueries in genetics for parallel processing.
            Split the query into task and context based on tags.
            Based on the context, ask relevant questions that help achieve the task. Make sure the subquestions make fully sense alone. If a marker or trait is shared between the subqueries, make sure to mention it explicitly in each subquery. There should be no words such as "this", "that" or "those" in the subqueries. The goal is to have subquestions that do not have any implicit relationships so that they can be processed separately.
            Return only the subquestions.
            Return strictly a JSON list of strings, nothing else.
            <|im_end|>
            <|im_start|>user
            Query:
            Task: Identify traits with a lod score > 3.0 for the marker Rsm10000011643. Tell me what this marker is involved in biology.
            Context: A trait name should contain strings like GWA, GEMMA or BXDPublish. The goal is to extract what we know in biology on the marker previously mentioned and link it to the traits identified. 
            Result:
            <|im_end|>
            <|im_start|>assistant
            ["What BXDPublish, GWA or GEMMA traits  have a lod score > 3.0 at Rsm10000011643?", "What is Rsm10000011643 involved in biology?"]
            <|im_end|>
            <|im_start|>user
            Query:
            Task: Identify traits with a lod score > 3.0 for the marker Rsm10000011643. Find what those traits are.
            Context: A trait name should contain strings like GWA, GEMMA or BXDPublish. The goal is to explain the association to the marker given the trait biology.
            Result:
            <|im_end|>
            <|im_start|>assistant
            ["What BXDPublish, GWA or GEMMA traits  have a lod score > 3.0 at Rsm10000011643?", "What are BXDPublish, GWA or GEMMA traits involved in biology?"]
            <|im_end|>
            <|im_start|>user
            Query:
            {query}
            Result:
            <|im_end|>
            <|im_start|>assistant"""

finalize_prompt = f"""
            <|im_start|>system
            You are an experienced geneticist. Given the subqueries and corresponding answers, generate a comprehensive explanation to address the query using all information provided.
            Ensure the response is insightful, concise, and draws logical inferences where possible.
            Do not modify entities names such as trait and marker.            
            Make sure to link based on what is common in the answers.
            Provide only the story, nothing else.
            Do not repeat answers. Use only 200 words max.
            <|im_end|>
            <|im_start|>user
            Query:
            Identify two traits related to diabetes.
            Compare their lod scores at Rsm149505.
            Subqueries:
            ["Identify two traits related to diabetes",
            "Compare lod scores of same traits at Rsm149505"]
            Answers:
            ["Traits A and B are related to diabetes", \
            "The lod score at Rsm149505 is 2.3 and 3.4 for trait A and B"]
            Conclusion:
            <|im_end|>
            <|im_start|>assistant
            Traits A and B are related to diabetes and have a lod score of\
            2.3 and 3.4 at Rsm149505. The two traits could interact via a\
            gene close to the marker Rsm149505.
            <|im_end|>
            <|im_start|>user
            Query:
            {query}
            Subqueries:
            {subqueries}
            Answers:
            {answers}
            Conclusion:
            <|im_end|>
            <|im_start|>assistant"""
