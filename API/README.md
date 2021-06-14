# API

The Pict2Text2 API is a Flask API, used to connect the two model (YOLO and One-shot) so that that they could be used as independent services and from the website.

Requirements:
 - python 3.8

Execute the following commands:
1. virtualenv venv
2. source venv/bin/activate
3. pip install -r requirements
  
## API structure

As the API uses both YOLO with our custom configuration and One-Shot, both othe the models were included into the directory of the API when executed. The One-Shot model is also included in the repository under the directory ***/routers/execute_one_shot*** but the YOLO was not as it was the clone darknet project with our configuration.

The structure of the API is:

API
  - routers
    - convert_jpg_to_png --> a copy of the scripy which changes the image format from [One-Shot model](https://github.com/NILGroup/TFG-2021-Pict2Text2.0/tree/master/One-shot/one-shot-model)
    - execute_one_shot --> a copy of the testing classes, weights, loading_images.py, siamese_network.py and execute_one_shot.py from [One-Shot model](https://github.com/NILGroup/TFG-2021-Pict2Text2.0/tree/master/One-shot/one-shot-model)
    - classify_pictograms.py --> the endpoints which execute the One-shot model
  -  static --> all the static components the API requires
  -  templates --> the views of the website
  -  YOLO --> a copy of [YOLO](https://github.com/NILGroup/TFG-2021-Pict2Text2.0/tree/master/YOLO) with cloned darknet, Makefile configured to use CPU, our custom configuration, build and capable to make predictions
    - /t --> a subfolder used by the API where the temporaty files are save after they are upload, in a separate directory. Important for both of the models!!!
    - script.sh --> a shell script used by the API to change the ***obj.data*** configuration file during execution pointing the names to the ***classes.txt*** path
  - app.py --> the main file from where the API starts, with the two endpoints for pictograms detection, yolo execution and the mapped two endpoints for classification
  - crop_bounding_boxes.py --> a python script to crop the pictograms using theirbounding boxes from the predicted image from the YOLO model
  - Utils.py --> the function used to create file names in md5 hashing. Used for unique identification when saving uploaded files.

#### Transforming images from ***jpg*** to ***png*** 
To transform the images from ***jpg*** to ***png*** use the script ***convert_to_png.py*** from the directory ***convert_jpg_to_png***. Executing the command:

```
python convert_to_png.py
```

The script will transform in ***png*** all images from the directory ***jpg_images*** and save them in ***png_images***. 

## Service
## How to execute?
