import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from PIL import Image

# Read in text files with the filenames and labels 
def read_txt_files(path):

    rows = []

    with open(path, encoding='UTF-8') as f:
        for line in f:

            line = line.strip()

            filename = line.split(",")[0]

            features = line.split(",")[1:]

            rows.append({'filename':filename, 'features':features})

        data = pd.DataFrame(rows)

        return data

test_data = read_txt_files('rover_image_classification/mer-labeled-data-set-v1.1.0/test-set-v1.1.0.txt')

train_data = read_txt_files('rover_image_classification/mer-labeled-data-set-v1.1.0/train-set-v1.1.0.txt')

val_data = read_txt_files('rover_image_classification/mer-labeled-data-set-v1.1.0/val-set-v1.1.0.txt')

# Around 30 rows in the train data are not in the images so will drop these rows doesnt seem to be the same in val and test data so will leave as is there
image_dir = Path('rover_image_classification/mer-labeled-data-set-v1.1.0/images')

folder_files = []
for f in image_dir.iterdir():

    folder_files.append(f.name)

folder_files = pd.DataFrame(folder_files,columns=['filename'])

drop_train_data = train_data[~train_data['filename'].isin(folder_files['filename'])].index

train_data.drop(drop_train_data,inplace=True)

print(len(val_data[~val_data['filename'].isin(folder_files['filename'])]))
print(len(test_data[~test_data['filename'].isin(folder_files['filename'])]))

# Loading images and returning arrays of the image data 
def load_images(path, size = (244,244)):

    image = Image.open(image_dir/path).convert('RGB').resize(size=size)

    return np.array(image)


train_data['images'] = train_data['filename'].apply(load_images)

test_data['images'] = test_data['filename'].apply(load_images)

val_data['images'] = val_data['filename'].apply(load_images)

original_train_images = train_data[train_data['filename'].str.endswith('img.jpg')]
original_test_images = test_data[test_data['filename'].str.endswith('img.jpg')]
original_val_images = val_data[val_data['filename'].str.endswith('img.jpg')]