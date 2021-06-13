import os, fnmatch
from PIL import Image

def convert_images_from_folder(folder = "jpg_images"):
    if not os.path.exists('png_images'):
        os.makedirs('png_images')
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        name_img_png = filename
        if os.path.splitext(filename)[1] != 'png':
            name_img_png = os.path.splitext(filename)[0] + '.png'
        img.save(os.path.join('png_images', name_img_png))


#convert_images_from_folder()

def convert_images(folder, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).resize((400, 400))
        name_img_png = filename
        if os.path.splitext(filename)[1] != 'png':
            name_img_png = os.path.splitext(filename)[0] + '.png'
        img.save(os.path.join(save_directory, name_img_png))

    print("Converting images done.")


#convert_images()