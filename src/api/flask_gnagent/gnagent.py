"""
This is the Flask endpoint of the system for integration into GN
Not used in this setup
"""

import os
import sys

from flask import Blueprint, jsonify, request

AGENT_PATH = "../../agent/"
sys.path.append(os.path.abspath(AGENT_PATH))

from agent import main

gnagent = Blueprint("gnagent", __name__)


@gnagent.route("/ask", methods=["POST"])
def ask_question():
    question = request.args.get("query", "")
    if not question:
        return jsonify({"error": "Question is missing"}), 400
    answer = main(question)
    return {
        "question": question,
        "answer": answer,
    }
