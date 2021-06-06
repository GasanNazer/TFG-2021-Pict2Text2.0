from flask import Flask, render_template, request, jsonify
from PIL import Image
import os
import io

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route("/detect_pictograms", methods=["POST"])
def predict():
    file = request.files['file']
    # Read the image via file.stream
    img = Image.open(file.stream)

    # save image
    path = app.root_path
    path = f"{path}/YOLO/darknet"
    path_config = f"{path}/pict2text_config/"

    img.save(f"{path_config}1.jpg")

    # execute YOLO
    return execute_yolo(file)

    #return jsonify({'msg': 'success', 'size': [img.width, img.height]})




def execute_yolo(picture_file = "../test/two_pictograms_test_close.jpg"):
    print("Executing prediction")
    path = app.root_path
    #command = f"cd {path}/YOLO/darknet; ls;"
    #command = f"cd {path}/YOLO/darknet; {path}/YOLO/darknet/darknet detector test {path}/YOLO/obj.data {path}/YOLO/yolov4-obj-test.cfg -ext_output {path}/YOLO/yolov4-obj_best.weights {picture_file} -thresh 0.1 > output.txt;" #-dont_show
    #os.system(command)
    #command = f"{path}/YOLO/darknet detector"

    path = f"{path}/YOLO/darknet"
    path_config = f"{path}/pict2text_config/"

    # create a script and execute it
    os.system(f"{path_config}script.sh {path_config}")
    os.system(f"{path}/darknet detector test {path_config}obj.data {path_config}yolov4-obj-test.cfg -ext_output {path_config}yolov4-obj_best.weights {path_config}1.jpg -thresh 0.1 > {path_config}output.txt -dont_show;")
    os.system(f"mv ./predictions.jpg {path_config}predictions.jpg")
    print("Finished")

    return "done"


