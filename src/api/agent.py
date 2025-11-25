"""
Api endpoint for GNAgent
To use:
curl -X POST -H "Content-Type: application/json" -d '{"query": <query>}' http://localhost:5000/
Pass query in <query>
"""

import asyncio

from flask import Blueprint, jsonify, request
from gnagent.agent import main

agent = Blueprint("agent", __name__)


@agent.route("/", methods=["POST"])
def ask_question():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is missing"}), 400
    answer = asyncio.run(main(query))
    return (
        jsonify(
            {
                "query": query,
                "answer": answer,
            }
        ),
        200,
    )
