import os
import cv2
import requests as req
import face_recognition as fr
from ast import literal_eval
from flask import Flask, make_response, jsonify, request
from os.path import join, dirname
from dotenv import load_dotenv

app = Flask(__name__)


dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)

if not os.path.isdir("img"):
    os.mkdir("img")


@app.route("/", methods=["GET"])
def status():
    return make_response(jsonify({ "status": "Ok" }), 200)


def download_image(image_remote_path, file_name):
    api_url = os.environ.get("API_URL")
    image_url = api_url + image_remote_path

    image_local_path = "./img/{0}".format(file_name)
    image_url = literal_eval(repr(image_url).replace("\\", "/"))

    image = open(image_local_path, "wb")
    response = req.get(image_url)

    image.write(response.content)
    image.close()
    return image_local_path


@app.route("/faces/compare", methods=["POST"])
def compare():
    json = request.get_json()

    try:
        registered_image = json["registeredImage"]
        image_compare = json["imageCompare"]
    except:
        return make_response(jsonify({ "Error": "Dados Inválidos" }), 400)


    registered_image_path = download_image(registered_image, "registered_image.{0}".format(registered_image.split(".")[-1]))
    image_compare_path = download_image(image_compare, "compare_image.{0}".format(image_compare.split(".")[-1]))

    image = fr.load_image_file(registered_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image2 = fr.load_image_file(image_compare_path)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    encode_image = fr.face_encodings(image)[0]
    encode_image2 = fr.face_encodings(image2)[0]

    is_equal = fr.compare_faces([encode_image], encode_image2)

    os.remove(registered_image_path)
    os.remove(image_compare_path)

    return make_response(jsonify({ "isEqual": bool(is_equal[0]) }), 200)


if __name__ == "__main__":
    app.run()