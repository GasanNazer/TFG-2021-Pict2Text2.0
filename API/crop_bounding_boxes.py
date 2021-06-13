import re
import cv2

def crop_bounding_boxes(directory_path):
    print(directory_path)
    directory_name = directory_path.split("/")
    directory_name = directory_name[len(directory_name) - 2]
    path = directory_path + f"{directory_name}_output.txt"

    print(f"path: {path}")

    myfile = open(path,'r')
    lines = myfile.readlines()
    pattern= "pictogram:"

    detected_pictograms = []

    for line in lines:
      if re.search(pattern,line):
        coordinates = re.findall("-?\d+", line.split("(")[1].split(")")[0])
        detected_pictogram = {"left_x" : int(coordinates[0]),
                              "top_y" : int(coordinates[1]),
                              "width" : int(coordinates[2]),
                              "height" : int(coordinates[3])}
        detected_pictograms.append(detected_pictogram)

    crop_and_save_detections(detected_pictograms, directory_path)

def crop_and_save_detections(detected_pictograms, directory_path):
    directory_name = directory_path.split("/")
    directory_name = directory_name[len(directory_name) - 2]
    image_file = f"{directory_path}{directory_name}.jpg"

    img = cv2.imread(image_file)

    for i in range(len(detected_pictograms)):
        y_min = abs(detected_pictograms[i].get("top_y"))
        y_max = abs(y_min + detected_pictograms[i].get("height"))
        x_min = abs(detected_pictograms[i].get("left_x"))
        x_max = abs(x_min + detected_pictograms[i].get("width"))

        crop_img = img[y_min:y_max, x_min:x_max]
        cv2.imwrite(f"{directory_path}{directory_name}_cropped_{i}.jpg", crop_img)

    print("Cropping finised")

