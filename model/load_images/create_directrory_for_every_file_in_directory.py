import os

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_directory_for_every_file_of_directory(folder = "pictograms_test"):
    for file in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, file)):
            folder_name = file.split("-")[0]
            make_directory(os.path.join(folder, folder_name))
            os.rename(os.path.join(folder, file), os.path.join(folder, os.path.join(folder_name, file)))

            #os.removedirs(file.split("-")[0])


create_directory_for_every_file_of_directory("pictograms_test")