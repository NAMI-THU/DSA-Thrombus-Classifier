import json

import flask
from flask import Flask, request
import Classificator

app = Flask(__name__)

classificator = Classificator.Classificator()


@app.route("/AiService/PreloadModels")
def prepare_models_requested():
    classificator.load_models()
    return flask.Response(status=200)


@app.route("/AiService/Classification", methods=["POST"])
def classification_requested():
    content = request.get_json()
    path_frontal = content["PathFrontal"]
    path_lateral = content["PathLateral"]
    activation_l, activation_f = classificator.do_classification(path_frontal, path_lateral)
    result = {'OutputFrontal': activation_f, 'OutputLateral': activation_l}
    return json.dumps(result)
