import datetime
import hashlib


def filename_hashing(filename):
    name = filename + "_" + str(datetime.datetime.now())
    print(name)
    result = hashlib.md5(name.encode()).hexdigest()
    print(result)

    return result