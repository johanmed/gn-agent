"""Main script of the API"""

from flask import Flask

from agent import agent as agent_blueprint

app = Flask(__name__)
app.register_blueprint(agent_blueprint)
