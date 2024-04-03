import numpy as np
from keras.layers import Input, Dense, Activation, ZeroPadding2D, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.preprocessing import image as keras_image
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import keras.backend as K
import os

import streamlit as st

#######################################################################################################################
modelSavePath = 'my_model3.h5'
numOfTestPoints = 2
batchSize = 16
numOfEpoches = 10
#######################################################################################################################

classes = ["Benign", "InSitu", "Invasive", "Normal"]

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def defModel(input_shape):
    X_input = Input(input_shape)

    X = Conv2D(16, (3, 3), strides=(1, 1))(X_input)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=3)(X)

    X = Conv2D(32, (3, 3), strides=(1, 1))(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=2)(X)

    X = Conv2D(64, (2, 2), strides=(1, 1))(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D(padding=(2, 2))(X)
    X = MaxPooling2D((2, 2), strides=2)(X)

    X = Conv2D(64, (2, 2), strides=(1, 1))(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D(padding=(2, 2))(X)
    X = MaxPooling2D((3, 3), strides=3)(X)

    X = Conv2D(32, (3, 3), strides=(1, 1))(X)
    X = Activation('relu')(X)

    X = Flatten()(X)
    X = Dense(256, activation='relu')(X)
    X = Dense(128, activation='relu')(X)
    X = Dense(len(classes), activation='softmax')(X)

    model = Model(inputs=X_input, outputs=X, name='Model')
    return model

@st.cache(allow_output_mutation=True)
def load_model_weights(model_path):
    model = defModel((512, 512, 3))
    model.load_weights(model_path)
    return model

def preprocess_image(img):
    img_resized = img.resize((512, 512))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict(img, model):
    x = img
    softMaxPred = model.predict(x)
    probs = softmaxToProbs(softMaxPred)
    return probs

def softmaxToProbs(soft):
    z_exp = [np.math.exp(i) for i in soft[0]]
    sum_z_exp = sum(z_exp)
    return [(i / sum_z_exp) * 100 for i in z_exp]

def predictImage(img, model):
    compProbs = [0] * len(classes)

    for i, crop in enumerate(img):
        st.write("Prediction for Crop " + str(i + 1) + ":\n")
        probs = predict(crop, model)

        for j, prob in enumerate(probs):
            st.write(f"{classes[j]} : {prob:.4f}%")
            compProbs[j] += prob

    # st.write("\n\nAverage Prediction Across All Crops:\n")
    max_prob_idx = np.argmax(compProbs)
    predicted_class = classes[max_prob_idx]
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Probability: {compProbs[max_prob_idx] / len(img):.4f}%")

    # Provide descriptions for each class
    class_descriptions = {
        "Benign": "Benign tumors are non-cancerous growths that do not spread to other parts of the body.",
        "InSitu": "In situ tumors are localized cancers that have not spread beyond their site of origin.",
        "Invasive": "Invasive tumors have the ability to spread to surrounding tissues or distant parts of the body.",
        "Normal": "Normal tissue without any signs of cancerous growth."
    }
    st.write(f"\n\n{predicted_class} Description:\n")
    st.write(class_descriptions.get(predicted_class, "No description available."))


def main():
    st.title("Cancer Detection")

    option = st.selectbox("Choose an option:", ["Test with your own custom image", "Test with sample image in the code"])

    if option == "Test with your own custom image":
        st.write("Upload your custom image:")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("")
            if st.button("Predict"):
                img = preprocess_image(image)
                model = load_model_weights(modelSavePath)
                predictImage([img], model)

    elif option == "Test with sample image in the code":
        st.write("Choose a sample image:")
        sample_images = ["benign1.jpg", "benign2.jpg", "InSitu1.jpg", "InSitu2.jpg", "Invasive1.jpg", "Invasive2.jpg", "Normal1.jpg", "Normal2.jpg"]
        selected_image = st.selectbox("Select sample image:", sample_images)
        image_path = os.path.join("sample_images", selected_image)
        image = Image.open(image_path)
        st.image(image, caption='Selected Sample Image', use_column_width=True)
        st.write("")
        if st.button("Predict"):
            img = preprocess_image(image)
            model = load_model_weights(modelSavePath)
            predictImage([img], model)

if __name__ == "__main__":
    main()
