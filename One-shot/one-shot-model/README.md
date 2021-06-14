# Training and testing the One-shot model

Requirements:
 - fetched pictograms using the [Fetching pictograms script](https://github.com/NILGroup/TFG-2021-Pict2Text2.0/tree/master/fetching_pictograms_tool)
 - python 3.8

Execute the following commands:
1. virtualenv venv
2. source venv/bin/activate
3. pip install -r requirements
- If Illegal instruction (core dumped) appears when executing the model, run the following command 
   ```
   pip install --upgrade --no-cache-dir tensorflow 
   ```
## Preparing the datasets

#### Transforming images from ***jpg*** to ***png*** 
To transform the images from ***jpg*** to ***png*** use the script ***convert_to_png.py*** from the directory ***convert_jpg_to_png***. Executing the command:

```
python convert_to_png.py
```

The script will transform in ***png*** all images from the directory ***jpg_images*** and save them in ***png_images***. 

#### Augmenting images

To augment images with the augmentation scripts from the directory ***image_augmentation*** you sould manually select some digital pictograms or pictures of pictograms and place them in a specific folder. For example **/pictograms** under a subfolder **/0**. Later modify the paramether of the function (***augment_images_from_folder("./pictograms", "photo")***) in the script ***augment_images.py*** pointing to the path of the above-described directory (**/pictograms**). The second paramether **photo** is a substring added to the name of the augmented images so that in case of multiple augmentation, like digiatal pictograms and pictures of pictorgrams, they could be easily distinguish.
To execute the 3 types of augmentations (color, brighness and rotation) use the following command:

```
python augment_images.py
```


### Training set
As described in the documentation of the project, the training set should contain augmented pictures of pictograms in ***png*** format, and the original digital pictogram. 

To augment the images use...



## Training the One-shot model

## Testing the One-shot model
