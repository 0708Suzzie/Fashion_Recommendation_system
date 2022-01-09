#importing required libraries 

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm  # this is to keep a progress on our training 
import pickle

#model to extract features from the dataset 
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([model, GlobalMaxPool2D()])

#extracting features using the above model 
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array  = image.img_to_array(img)
    expand_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img_array)
    result  = model.predict(preprocessed_img).flatten()
    normalized_result = result/norm(result)
    return normalized_result


#extracting features and storing them 
filenames = []

for file in os.listdir('images') :
    filenames.append(os.path.join('images', file))

feature_list = []

for file in tqdm(filenames) :
    feature_list.append(extract_features(file, model))


pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

