# GNAgent

## Description

**GNAgent** is an agentic system designed for [GeneNetwork](https://genenetwork.org/). It leverages reasoning abilities of Large Language Models (LLMs) to autonomously handle genomic tasks complex for humans. 

## Design

**GNAgent** combines 04 agents to provide insights into genomics of model organisms:
- *Supervisor*: digests task, delegates responsibilities to specialized agents and manages interaction between them
- *Planner*: plans series of steps to take to solve the task at hand
- *Reflector*: improves reasoning by providing thoughtful suggestions and critiques
- *Researcher*: extracts knowledge and addresses queries using a RAG architecture

## Configuration

All functionalities of **GNAgent** have been packaged for seamless experimentation and use. To get started, take the following steps:

1. Clone the repository
```bash
git clone https://github.com/johanmed/gn-agent.git
```

2. Install poetry
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install poetry
```

3. Install all dependencies
```bash
poetry install
```

4. Activate virtual environment
```bash
eval $(poetry env activate)
```

5. Navigate to source code
```bash
cd src/
```

6. Run the system
```bash
python3 agent.py
```

## Useful considerations
- This project is still under development. We are actively chasing bugs :)
- We support [DSpy](https://dspy.ai/) for portability, reliability and optimization. You can switch to any models including proprietary models.
- Be sure to not modify the files `pyproject.toml` and `poetry.lock`
