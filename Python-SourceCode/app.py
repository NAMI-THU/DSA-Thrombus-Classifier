import json

from flask import Flask, request
import doClassification

app = Flask(__name__)


@app.route("/AiService/Classification", methods=["POST"])
def classification_requested():
    content = request.get_json()
    path_frontal = content["PathFrontal"]
    path_lateral = content["PathLateral"]
    activation_l, activation_f = doClassification.do_classification(path_frontal, path_lateral)
    result = {'OutputFrontal': activation_f, 'OutputLateral': activation_l}
    return json.dumps(result)
