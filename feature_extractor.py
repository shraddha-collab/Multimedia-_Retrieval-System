from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalMaxPooling2D
import numpy as np


# Loading the pre-trained ResNet50 model
resnet_model = ResNet50(
    weights="imagenet",
    include_top=False
)


# Creating a model to extract image features
feature_model = Model(
    inputs=resnet_model.input,
    outputs=GlobalMaxPooling2D()(resnet_model.output)
)


# Function to get features from an image
def get_image_features(image_path):

    # Open image and resize it
    img = image.load_img(
        image_path,
        target_size=(224, 224)
    )

    # Convert image into numbers
    img = image.img_to_array(img)

    # Add batch size
    img = np.expand_dims(img, axis=0)

    # Prepare image for ResNet50
    img = preprocess_input(img)

    # Extract features
    features = feature_model.predict(
        img,
        verbose=0
    )

    # Convert features into a simple list
    features = features.flatten()

    # Normalize features for better comparison
    features = features / np.linalg.norm(features)

    return features
    