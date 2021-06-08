import base64
import datetime
import hashlib
import time

from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import os
import io

from werkzeug.utils import secure_filename, redirect

app = Flask(__name__)

#@app.route('/')
def hello_world():
    return render_template('index.html')

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
    print("Finished")

    return path_prediction


app.config['UPLOAD_FOLDER'] = app.root_path + "/YOLO/t"

@app.route('/')
@app.route('/upload')
def upload():
   return render_template('upload.html')

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


@app.route('/detect_pictograms_web', methods = ['GET', 'POST'])
def detect_pictogram_web():
   if request.method == 'POST':
      f = request.files['file']
      file_name = filename_hashing(f.filename) + "." + f.filename.split(".")[1]
      f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(file_name)))

      path = execute_yolo(file_name)

      time.sleep(5)

      image = f"{path}{file_name.split('.')[0]}_predictions.jpg"

      im = Image.open(image)
      data = io.BytesIO()
      im.save(data, "JPEG")
      encoded_img_data = base64.b64encode(data.getvalue())

      return render_template("result.html", img_data=encoded_img_data.decode('utf-8'))

   else:
       return redirect('/upload')



def filename_hashing(filename):
    name = filename + "_" + str(datetime.datetime.now())
    print(name)
    result = hashlib.md5(name.encode()).hexdigest()
    print(result)

    return result


