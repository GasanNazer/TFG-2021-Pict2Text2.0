import os
import shutil


def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_directory_for_every_file_of_directory(folder = "pictograms_test"):
    count = 0
    for file in os.listdir(folder):
        count += 1
        subdirectory = str(count // 4000)
        if os.path.isfile(os.path.join(folder, file)):
            folder_name = file.split("-")[0]
            make_directory(os.path.join(folder, folder_name))
            os.rename(os.path.join(folder, file), os.path.join(folder, os.path.join(folder_name, file)))

            make_directory(os.path.join(folder, subdirectory))
            origin_directory = os.path.join(folder, folder_name)
            target_directory = os.path.join(folder, os.path.join(os.path.join(subdirectory, folder_name)))
            shutil.move(origin_directory, target_directory)

            #os.removedirs(file.split("-")[0])


create_directory_for_every_file_of_directory("pictograms")