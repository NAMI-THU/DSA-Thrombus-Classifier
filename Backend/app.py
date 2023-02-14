import json
import os.path

import flask
import numpy as np
from flask import Flask, request

import Classificator

app = Flask(__name__)

classificator = Classificator.Classificator()


@app.route("/AiService/PreloadModels", methods=["POST"])
def prepare_models_requested():
    folder = request.get_json()["Directory"]
    print(f"Preloading of models requested... ({folder})")

    if classificator.check_models_already_loaded(folder):
        print("Models have already been loaded before.")
        return flask.Response(status=200)

    classificator.load_models(folder)
    print("Loading of models done.")
    return flask.Response(status=200)


@app.route("/AiService/Classification", methods=["POST"])
def classification_requested():
    print("Classification requested...")
    content = request.get_json()
    model_frontal = content["ModelFrontal"]
    model_lateral = content["ModelLateral"]
    image_f = content["PathFrontal"]
    image_l = content["PathLateral"]
    print(f"Classification running...")
    activations_f, activations_l, estimates_f, estimates_l = classificator.do_classification(image_f,
                                                                                             image_l,
                                                                                             model_frontal,
                                                                                             model_lateral)
    result = {'OutputFrontal': activations_f, 'OutputLateral': activations_l}
    print(f"Classification done. Results: {activations_f} | {activations_l}")
    return json.dumps(result)


@app.route("/AiService/LoadImages", methods=["POST"])
def load_images_requested():
    print(f"Loading of images requested...")
    content = request.get_json()
    image_f = content["PathFrontal"]
    image_l = content["PathLateral"]
    with_normalization = content["Normalized"]
    print(f"Loading images {image_f} and {image_l}")
    if not image_f or not os.path.exists(image_f) or not image_l or not os.path.exists(image_l):
        print("At least one of the requested paths does not exist.")
        return flask.Response(status=400)
    dict = classificator.prepare_images(image_f, image_l, with_normalization)
    print("Loading done, packaging now...")

    return json.dumps(dict, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)