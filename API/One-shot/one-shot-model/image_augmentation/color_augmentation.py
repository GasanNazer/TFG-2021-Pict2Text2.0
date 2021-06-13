import os
from random import random
from PIL import Image, ImageDraw
from keras_preprocessing.image import load_img


def generate_images_with_augmented_color(file_path, filename, image_folder, augmentation_type, samples):
    image = load_img(file_path, color_mode="rgba")
    input_pixels = image.load()

    count = 1

    while count < samples:
        rand_r = random()
        rand_b = random()
        rand_g = random()

        # Create output image
        output_image = Image.new("RGBA", image.size)
        draw = ImageDraw.Draw(output_image)

        # Generate image
        for x in range(output_image.width):
            for y in range(output_image.height):
                r, g, b, a = input_pixels[x, y]
                r = int(r * 5 * rand_r + 50)
                g = int(g * 5 * rand_g * rand_g + 50)
                b = int(b * 5 * rand_b + 50)
                draw.point((x, y), (r, g, b, a))

        output_image.save(os.path.join(image_folder, f'{filename.split(".")[0]}_{augmentation_type}_{count}.png'))
        count += 1