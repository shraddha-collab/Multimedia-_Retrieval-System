import os
import pickle
from tqdm import tqdm
from feature_extractor import get_image_features


# Dataset folder path
dataset_folder = "dataset/images/seg_train"

# Lists to store image features and image paths
embeddings = []
filenames = []


# Reading all images from dataset
for folder in os.listdir(dataset_folder):

    folder_path = os.path.join(dataset_folder, folder)

    if os.path.isdir(folder_path):

        for image_name in tqdm(os.listdir(folder_path)):

            image_path = os.path.join(folder_path, image_name)

            try:
                # Extract image features
                feature = get_image_features(image_path)

                # Store features and image path
                embeddings.append(feature)
                filenames.append(image_path)

            except Exception as error:
                print("Error in image:", image_path)
                print(error)


# Save image features
with open("embeddings.pkl", "wb") as file:
    pickle.dump(embeddings, file)


# Save image names/paths
with open("filenames.pkl", "wb") as file:
    pickle.dump(filenames, file)


print("Feature extraction completed!")
print("Embeddings and filenames files created successfully.")