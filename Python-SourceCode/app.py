import json
import os.path

import flask
from flask import Flask, request
import Classificator

app = Flask(__name__)

classificator = Classificator.Classificator()


@app.route("/AiService/PreloadModels")
def prepare_models_requested():
    print("Preloading of models requested...")
    if classificator.models_loaded:
        print("Models have already been loaded before.")
        return flask.Response(status=200)

    classificator.load_models()
    print("Loading of models done.")
    return flask.Response(status=200)


@app.route("/AiService/Classification", methods=["POST"])
def classification_requested():
    print("Classification requested...")
    content = request.get_json()
    path_frontal = content["PathFrontal"]
    path_lateral = content["PathLateral"]
    if not path_frontal or not os.path.exists(path_frontal) or not path_lateral or not os.path.exists(path_lateral):
        print("At least one of the requested paths does not exist.")
        return flask.Response(status=400)
    activations_f, activations_l, estimates_f, estimates_l = classificator.do_classification(path_frontal, path_lateral)
    result = {'OutputFrontal': activations_f, 'OutputLateral': activations_l}
    print("Classification done.")
    return json.dumps(result)
