import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r'fabricdefect.hdf5', compile=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def perform_classification(image, model):
    image_shape = (224, 224)
    image = ImageOps.fit(image, image_shape, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshaped = img[np.newaxis, ..., np.newaxis]
    img_reshaped = img_reshaped.astype('float32') / 255.0
    classifier = model.predict(img_reshaped)
    return classifier

def main():
    st.title("Fabric Defect Classification using CNN")

    model = load_model()

    uploaded_file = st.file_uploader("Please upload a Fabric Detection File", type=["jpg", "png"])

    if uploaded_file is None:
        st.text("Please upload an image")
    else:
        # Placeholder for displaying the output
        output_placeholder = st.empty()

        # Process the image and get the classification
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        classification = perform_classification(image, model)
        classes = ['Holes', 'Horizontal', 'Vertical']
        predicted_class = np.argmax(classification)
        classification_max = classification[0][predicted_class]

        # Display the classification in the placeholder
        output_placeholder.write(classification_max)

        result = f"The defect in the fabric is of type: {classes[predicted_class]}"
        st.success(result)

if __name__ == "__main__":
    main()
