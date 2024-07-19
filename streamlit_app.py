import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

# Load the saved EfficientNet model
model_saved = tf.keras.models.load_model('models/corneff_20240719_93p75.h5')

# List of class labels
all_labels = ['corn_gray_leaf_spot', 'corn_healthy', 'corn_northern_leaf_blight', 'corn_rust']

def preprocess_image(img_path, target_size):
    try:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, target_size)
            img = img.astype('float32')  # No need for normalization here
            img = preprocess_input(img)  # Use EfficientNet's preprocessing
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            return img
        else:
            print(f"Error: Unable to read image at {img_path}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    st.title("Corn Disease Classification")

    st.write("Upload an image of a corn leaf to classify its health status:")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary file path
        img_path = 'tmp/temp_image.jpg'
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display the image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image and make predictions
        input_shape = model_saved.input_shape[1:3]  # (256, 256)
        pre_img = preprocess_image(img_path, input_shape)
        if pre_img is not None:
            predict = model_saved.predict(pre_img)
            class_index = np.argmax(predict)  # Find the index of the highest probability
            class_name = all_labels[class_index]  # Get the class label
            st.write(f'Prediction: {predict}')
            st.write(f'Class: {class_name}')
        else:
            st.write('Could not process the image.')

if __name__ == "__main__":
    main()
