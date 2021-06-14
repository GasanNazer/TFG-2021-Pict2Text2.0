# Training and testing the One-shot model

Requirements:
 - fetched pictograms using the [Fetching pictograms script](https://github.com/NILGroup/TFG-2021-Pict2Text2.0/tree/master/fetching_pictograms_tool)
 - python 3.8

Execute the following commands:
1. Create virtual environment
```
virtualenv venv
```
2. Activate virtual environment 
```
source venv/bin/activate
```
3. Install requirements
```
pip install -r requirements
```
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

To augment images with the augmentation scripts from the directory ***image_augmentation*** you sould manually select some digital pictograms or pictures of pictograms and place them in a specific folder. For example subfolder **/0** of directory **/pictograms**. Later modify the paramether of the function (***augment_images_from_folder("./pictograms", "photo")***) in the script ***augment_images.py*** pointing to the path of the above-described directory (**/pictograms**). The second paramether **photo** is a substring added to the name of the augmented images so that in case of multiple augmentation, like digiatal pictograms and pictures of pictorgrams, they could be easily distinguish.
To execute the 3 types of augmentations (color, brighness and rotation) use the following command:

```
python augment_images.py
```


### Training set
As described in the documentation of the project, the training set should contain augmented pictures of pictograms in ***png*** format, and the original digital pictogram. 

A sample of the training dataset could be seen in the directory ****pictograms_train***. In it the pictures of pictograms are augmented using the above-described augmentation process and later the digital pictogram is manually included in each class directory.

### Validation/Test sets
The validation set and the test set consist of two parts: digital pictograms or pictures of pictograms (in png format) you want to validate/test with, and a bigger set of digital pictograms to predict against.

A sample of the validation set could be seen in the directories ***pictograms_val*** and ***pictograms_val_digital***.
The pictures of pictograms in ***pictograms_val*** should have the same format as the original digital pictogram: <id>-<word>.png

To prepare the data in ***pictograms_val_digital*** some digital pictograms were selected from the fetched pictograms from ARASAAC and the following script was executed ***create_directrory_for_every_file_in_directory.py***.
Make sure the provided path in the script coresponds to the location of the directory with the data.

```
python create_directrory_for_every_file_in_directory.py
```
All of the above is aplicable to the test set.

## Training and validating the One-shot model
 In the scri- ***main.py*** the follwing attributes should be taken into consideration:
 - **n_iter** - the number of iterations the model is going to execute to train itself 
 - **evaluate_every** - interval for evaluating on one-shot tasks with the validation set, calculates accuracy and saves the weights in the folder ***weights***
 - **batch_size** - the number of pairs taken in each iteration. It should not surpass the number of classes in the training set and it should be bigger than 1.
 
 The following command trains and validates the model:
 
 ```
 python main.py
 ```

## Testing the One-shot model
As we have mentioned previously, the weights obtained from training the algorithm are being stored in folder ***weights***, named as the number of iterations the model was executed with to obtain them. In order to test the algorithm with specific weights, the number indicated in the name of the file should be provided in the list (***weights***) in the script **model.py**.
 
To test the model execute the command:
```
python model.py 
```
 
If it is not required to have the accuracy of the testing set but only the prediction part, the script ***execute_one_shot.py*** could be used.
 
