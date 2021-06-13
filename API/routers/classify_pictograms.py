import os

from flask import request
from werkzeug.utils import secure_filename

from Utils import filename_hashing
from flask import current_app
import routers.convert_jpg_to_png.convert_to_png as images_converter

from routers.execute_one_shot.execute_one_shot import make_prediction_one_shot

def classify_pictograms():
    print(current_app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':
        f = request.files['file']
        file_name = filename_hashing(f.filename) + "." + f.filename.split(".")[1]
        f.save(os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(file_name)))

        if f.content_type == "image/jpeg":
            path_image = os.path.join(current_app.config['UPLOAD_FOLDER'], file_name)
            path_jpg_images = os.path.join(current_app.root_path, "routers/convert_jpg_to_png/jpg_images")

            if not os.path.exists(path_jpg_images):
                os.makedirs(path_jpg_images)

            os.system(f"mv {path_image} {path_jpg_images}")
            images_converter.convert_images(path_jpg_images, os.path.join(current_app.root_path, "routers/convert_jpg_to_png/png_images"))
        elif f.content_type == "image/png":
            pass
        else:
            return "Incorrect file type"

        # execute classification
        pictograms_folder = os.path.join(current_app.root_path, "routers/convert_jpg_to_png/png_images")
        test_pictograms_folder = os.path.join(current_app.root_path, "routers/execute_one_shot/pictograms")

        predictions = make_prediction_one_shot(pictograms_folder, test_pictograms_folder, os.path.join(current_app.root_path, "routers/execute_one_shot/weights"))

        os.system(f"rm -r {path_jpg_images}")
        os.system(f"rm -r {pictograms_folder}")

        return {"predictions" : str(predictions)}

def classify_pictograms_web(pictograms_folder):
    predictions = None

    for filename in os.listdir(pictograms_folder):
        if "_cropped_" in filename:
            path_jpg_images = os.path.join(current_app.root_path, "routers/convert_jpg_to_png/jpg_images")

            if not os.path.exists(path_jpg_images):
                os.makedirs(path_jpg_images)

            os.system(f"cp {pictograms_folder}{filename} {path_jpg_images}")
            images_converter.convert_images(path_jpg_images,
                                    os.path.join(current_app.root_path, "routers/convert_jpg_to_png/png_images"))
    os.system(f"rm -r {path_jpg_images}")
    os.system(f"mv {os.path.join(current_app.root_path, 'routers/convert_jpg_to_png/png_images')} {pictograms_folder}")

    # execute classification
    pictograms_folder = os.path.join(pictograms_folder, "png_images")
    test_pictograms_folder = os.path.join(current_app.root_path, "routers/execute_one_shot/pictograms")

    predictions = make_prediction_one_shot(pictograms_folder, test_pictograms_folder, os.path.join(current_app.root_path, "routers/execute_one_shot/weights"))

    return predictions

