import os
from PIL import Image

def convert_format(path = 'Split_Data/Airliners/Test', old_format = 'jpg', new_format = 'png'):
    for file in os.listdir(path):
        print(file)
        for pict in os.listdir(path + '/' + file):
            if pict[-len(old_format):] == old_format:
                im = Image.open(path + '/' + file + '/' + pict)
                im.save(path + '/' + file + '/' + pict[:-len(old_format)] + new_format)
                os.remove(path + '/' + file + '/' + pict)
