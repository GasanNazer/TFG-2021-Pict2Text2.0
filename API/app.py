import base64
import json
import time

from flask import Flask, render_template, request, send_file, url_for
from PIL import Image
import os
import io

from werkzeug.utils import secure_filename, redirect

import routers.classify_pictograms as pictogram_classification
import crop_bounding_boxes as cropper
from Utils import filename_hashing
from routers.classify_pictograms import classify_pictograms_web
import requests

app = Flask(__name__)

def execute_yolo(filename):
    print("Executing prediction")
    path = app.root_path

    path = f"{path}/YOLO/darknet"
    path_config = f"{app.root_path}/YOLO/"
    path_prediction = f"{path_config}t/{filename.split('.')[0]}/"
    prediction_name = f"{path_prediction}{filename.split('.')[0]}"

    # create a script and execute it
    os.system(f"mkdir {path_prediction}; mv {path_config}t/{filename} {path_prediction}/{filename}")
    os.system(f"{path_config}script.sh {path_config}")
    os.system(f"cd {path}; ./darknet detector test {path_config}obj.data {path_config}yolov4-obj-test.cfg -ext_output {path_config}yolov4-obj_best.weights {path_prediction}{filename} -thresh 0.1 > {prediction_name}_output.txt -dont_show;")
    os.system(f"mv {path}/predictions.jpg {prediction_name}_predictions.jpg")

    return path_prediction


app.config['UPLOAD_FOLDER'] = app.root_path + "/YOLO/t"

@app.route('/')
def upload():
   return redirect('/uploadImage')

@app.route('/detect_pictograms', methods = ['GET', 'POST'])
def detect_pictogram():
   if request.method == 'POST':
      f = request.files['file']
      file_name = filename_hashing(f.filename) + "." + f.filename.split(".")[1]
      f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(file_name)))

      path = execute_yolo(file_name)

      image = f"{path}{file_name.split('.')[0]}_predictions.jpg"

      return send_file(image,
          mimetype='image/jpeg',
          as_attachment=True,
          attachment_filename=f'{file_name}_predictions.jpg')


def execute_models(f, file_name):
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file_name)))

    path = execute_yolo(file_name)

    time.sleep(5)

    image = f"{path}{file_name.split('.')[0]}_predictions.jpg"

    cropper.crop_bounding_boxes(path)

    # executing the One-shot model
    predictions = classify_pictograms_web(path)

    return predictions

# routers /classify_pictograms
app.add_url_rule('/classify_pictograms', '/classify_pictograms', pictogram_classification.classify_pictograms, methods=["POST"])
app.add_url_rule('/classify_pictograms_web', '/classify_pictograms_web', pictogram_classification.classify_pictograms_web)

@app.route('/uploadImage', methods = ['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = None
        file_name = None
        file_hashing = None
        if request.form.get("flexCheckDefault") is None:
            uploaded_file = request.files['file1']
            if uploaded_file.filename != '':
                file = uploaded_file
                file_hashing = filename_hashing(file.filename)
                file_name = file_hashing + "." + file.filename.split(".")[1]
        else:
            path = request.form.get('inlineRadioOptions')
            if path != None:
                file = Image.open(app.root_path + path)
                file_n = file.filename.split("/")[-1]
                file_hashing = filename_hashing(file_n)
                file_name = file_hashing + "." + file_n.split(".")[1]
        if file is None:
            return redirect('/uploadImage')
        predictions = execute_models(file, file_name)

        path = app.root_path + "/YOLO/t/" + file_hashing
        number_of_cropped_images = 0

        for filename in os.listdir(path):
            if "_cropped_" in filename:
                number_of_cropped_images += 1


        return redirect(url_for('show_results', hashing=file_hashing, num=number_of_cropped_images, predictions={"predictions": predictions}))
    else:
        return render_template('upload_template.html')

def encrypt_image(path):
    im = Image.open(path)
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    encoded_img_data = encoded_img_data.decode('utf-8')

    return encoded_img_data

@app.route('/show_results')
def show_results():
    hashing = request.args.get('hashing', None)
    num = request.args.get('num', None)
    similarities = request.args.get('predictions', None)
    path = app.root_path + "/YOLO/t/" + hashing + "/" + hashing + "_predictions.jpg"

    cropped_img = []
    for i in range(0, int(num)):
        path_cropped_image = app.root_path + f"/YOLO/t/{hashing}/{hashing}_cropped_{i}.jpg"
        cropped_img.append(encrypt_image(path_cropped_image))

    encoded_img_data = encrypt_image(path)

    predictions = []

    similarities = similarities.replace("\'", "\"")
    similarities = json.loads(similarities)
    for similarity in similarities["predictions"]:
        predictions.append({"id" : int(similarity["id"]),
                            "word": call_ARASAAC(int(similarity["id"])),
                            "similarity": float(similarity["similarity"])})

    return render_template('results_template.html', num=int(num), original=encoded_img_data, cropped=cropped_img, predictions=predictions)


def call_ARASAAC(id_pictogram):
    pictograms_es = requests.get(f"https://api.arasaac.org/api/pictograms/{id_pictogram}/languages/es").json()
    word = pictograms_es["keywordsByLocale"]["es"][0]["keyword"]
    return word


