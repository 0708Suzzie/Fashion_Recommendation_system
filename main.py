#importing needed libraries 
import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import pickle
from numpy.linalg import norm

#loading the features and filenames 
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

#creating the model so we could extract features from the input file 
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([model, GlobalMaxPool2D()])

#saving the uploaded file
def save_upload_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f :
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


#extracting features from the uploaded file 
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


#this is our recommender based on KNN using euclidean distances from sklearn's nearestneighbor
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])
    return indices


st.title('Fashion Recommender System')

#uploading files 
uploaded_file = st.file_uploader("Choose an image")

#our main recommender applied 
if uploaded_file is not None :
    if save_upload_file(uploaded_file):
        #display file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        #extract features
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        #recommend
        indices = recommend(features, feature_list)
        #display
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])

    else:
        st.header("some error in file upload")
