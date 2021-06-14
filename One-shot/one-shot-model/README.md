# Training and testing the One-shot model

Requirements:
 - fetched pictograms using the [Fetching pictograms script](https://github.com/NILGroup/TFG-2021-Pict2Text2.0/tree/master/fetching_pictograms_tool)
 - python 3.8

Execute the following commands:
1. virtualenv venv
2. source venv/bin/activate
3. pip install -r requirements
4. python ./model.py
4a. If Illegal instruction (core dumped) appears after running the previous instruction, run the following command 
   pip install --upgrade --no-cache-dir tensorflow 
   
## Preparing the datasets
### Training set
As described in the documentation of the project, the training set should contain augmented, pictures of pictograms in ***png*** format, and the original digital pictogram. 

To transform the images from ***jpg*** to ***png*** use the script ****convert_to_png.py*** from the directory ***convert_jpg_to_png***. Executing the command:

```
python convert_to_png.py
```

The script will transform in ***png*** all images from the directory ***jpg_images*** and save them in ***png_images***. 



## Training the One-shot model

## Testing the One-shot model
