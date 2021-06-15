# Training and testing the YOLO model 
For both of these actions first clone the original YOLO project executing:
```
git clone https://github.com/AlexeyAB/darknet
```
## Training the YOLO model
As the darknet is a deep neural network, the trainging process requires computational resources. Having this into account, we recommend training the model in Google Colaboratory.
Upload the notebook (***YOLOv4.ipynb***) from the directory ***google_drive_configuration*** to your google drive and follow the steps from the notebook, using the configuration files and the dataset 
from the compressed file (***drive_configuration_and_dataset.zip***).

## Testing the YOLO model

- Modify the Makefile according to the way you want to execute the model (using GPU or CPU)
  - GPU
  - CUDNN
  - CUDNN_HALF
  - AVX
  - OPENMP
  - LIBSO
  
  Note: Consult the files ***Makefile*** and ***Makefile_test***.
  
- Build the darknet project executing:

```
make
```
- Make sure the ***obj.data*** points to the correct paths. They should be mapped with respect of the path from where the next command will be executed.
- Execute the YOLO model with custom configuration:
```
cd darknet; \ 
./darknet detector test {path_config}obj.data {path_config}yolov4-obj-test.cfg -ext_output {path_config}yolov4-obj_best.weights {filename} -thresh 0.1 > output.txt;
```

where:
- ***{path_config}*** - the path of where the custom configuration is located.
- ***{filename}*** - a picture to test against
  > example: ../test/single_pictogram_zoom_out_small_size.jpg



The file ***yolov4-obj_best.weights*** contains the best weights for our YOLO model obtained in the version 2 of our training.
