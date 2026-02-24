"""This module addresses better a query related to a specific task with GNAgent"""

from pathlib import Path

from adapter import QUERY, dspy_agent

task = input("Genomic task to perform: ")
prompt_path = f"prompts2/{task}.py"
example_path = f"examples/{task}"

if Path(prompt_path).exists():
    with open(prompt_path) as f:
        optimized_prompt = f.read()
        new_prompt = f"""
        You previously solved a task very well with the following instructions:
        {optimized_prompt}
        Use sensibly the same instructions to provide state-of-art answers to this new task:
        {QUERY}
        """
        logging.info("Using optimized prompt with GNAgent...")
        output = dspy_agent(query=new_prompt)
        output = output.get("answer")
        logging.info(f"System feedback: {output}")
        # Save queries and answers for reuse
        with open(f"{example_path}/optimized.csv", "a") as a:
            formatted_prompt = optimized_prompt.replace("\n", "   ")
            formatted_output = output.replace("\n", "   ")
            to_write = f"'{formatted_prompt}','{formatted_output}'\n"
            a.write(to_write)
else:
    output = dspy_agent(query=QUERY)
    output = output.get("answer")
    logging.warning("Using user supplied prompt...")
    logging.info(f"System feedback: {output}")
    # Save queries and answer for reuse
    with open(f"{example_path}/original.csv", "a") as a:
        formatted_query = QUERY.replace("\n", "   ")
        formatted_output = output.replace("\n", "   ")
        to_write = f"'{formatted_query}','{formatted_output}'\n"
        a.write(to_write)
