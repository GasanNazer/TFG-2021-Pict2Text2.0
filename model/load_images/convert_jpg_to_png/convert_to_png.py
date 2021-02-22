import os, fnmatch
from PIL import Image

def augment_images_from_folder(folder = "jpg_images"):
    if not os.path.exists('png_images'):
        os.makedirs('png_images')
    for filename in os.listdir(folder):
        os.path.join(folder, filename)
        img = Image.open(os.path.join(folder, filename))
        name_img_png = filename
        if os.path.splitext(filename)[1] != 'png':
            name_img_png = os.path.splitext(filename)[0] + '.png'
        img.save(os.path.join('png_images', name_img_png))


augment_images_from_folder()
