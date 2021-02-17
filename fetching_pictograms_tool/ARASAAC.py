import concurrent

import requests
import os
from os import listdir
from os.path import isfile, join
import shutil

from concurrent.futures import ThreadPoolExecutor

PICTOGRAM_SIZE = 500
#LIMIT_FETCHING = 40000 #fetch only 4000
BATCH = 0
DIRECTORY_NAME = "pictograms" + "/" + str(BATCH)

pictograms_es = requests.get("https://api.arasaac.org/api/pictograms/all/es").json()
pictograms_id_keywords = dict()

for pictogram in pictograms_es:
    pictograms_id_keywords[pictogram["_id"]] = pictogram["keywords"]

if os.path.exists("pictograms"):
    shutil.rmtree('pictograms')

def change_directory_if_needed():
    global DIRECTORY_NAME, BATCH

    only_files = []
    if os.path.exists(DIRECTORY_NAME):
        only_files = [f for f in listdir(DIRECTORY_NAME) if isfile(join(DIRECTORY_NAME, f))]

    if len(only_files) > 4000:
        BATCH += 1

    DIRECTORY_NAME = "pictograms" + "/" + str(BATCH)

    if not os.path.exists(DIRECTORY_NAME):
        os.makedirs(DIRECTORY_NAME)

#currently_fetched_files = 1

def fetch():
    change_directory_if_needed()
    if not os.path.exists(f"{DIRECTORY_NAME}/{picto_id}-{pictograms_id_keywords[picto_id][0]['keyword']}.png"):
        pic_keyword = pictograms_id_keywords[picto_id][0]['keyword']
        if "/" in pic_keyword or "\\" in pic_keyword:
            pic_keyword = pictograms_id_keywords[picto_id][1]['keyword'] if pictograms_id_keywords[picto_id][1]['keyword'] != None else "unknown_pictogram"
        image = requests.get(f"https://static.arasaac.org/pictograms/{picto_id}/{picto_id}_{PICTOGRAM_SIZE}.png").content

        f = open(f"{DIRECTORY_NAME}/{picto_id}-{pic_keyword}.png", "wb")
        f.write(image)
        f.close()

for picto_id in pictograms_id_keywords:
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        thread1 = executor.submit(fetch)
    #if currently_fetched_files > LIMIT_FETCHING:
    #    break
    #currently_fetched_files += 1


print("fetching done")