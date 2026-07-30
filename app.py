from feature_extractor import get_image_features
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import json
import bcrypt
import os
import pickle
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(
    page_title="Multimedia Information Retrieval System",
    layout="wide"
)


theme = st.sidebar.selectbox(
    " Select Theme",
    ["Pink", "Blue", "Green", "Purple", "Orange"]
)

themes = {
    "Pink": {
        "primary": "#FF69B4",
        "background": "#FFF5FA",
        "secondary": "#FFE4F0",
        "text": "#333333"
    },
    "Blue": {
        "primary": "#1E90FF",
        "background": "#F5FAFF",
        "secondary": "#DCEEFF",
        "text": "#222222"
    },
    "Green": {
        "primary": "#2E8B57",
        "background": "#F5FFF8",
        "secondary": "#DFF5E5",
        "text": "#222222"
    },
    "Purple": {
        "primary": "#8A2BE2",
        "background": "#FAF5FF",
        "secondary": "#E8D9FF",
        "text": "#222222"
    },
    "Orange": {
        "primary": "#FF8C00",
        "background": "#FFF8F0",
        "secondary": "#FFE4C4",
        "text": "#222222"
    }
}

selected = themes[theme]

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {selected["background"]};
        color: {selected["text"]};
    }}

    [data-testid="stSidebar"] {{
        background-color: {selected["secondary"]};
    }}

    .stButton button {{
        background-color: {selected["primary"]};
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


USER_FILE = "users.json"
with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

with open("filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

text_df = pd.read_csv("text_data.csv")

vectorizer = TfidfVectorizer()

text_vectors = vectorizer.fit_transform(
    text_df["description"] + " " + text_df["tags"]
)

if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({"usernames": {}}, f)


def load_users():
    with open(USER_FILE, "r") as f:
        return json.load(f)


def save_users(data):
    with open(USER_FILE, "w") as f:
        json.dump(data, f, indent=4)


def register_user(username, password):

    users = load_users()

    if username in users["usernames"]:
        return False

    hashed = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    users["usernames"][username] = {
        "password": hashed
    }

    save_users(users)

    return True


def verify_user(username, password):

    users = load_users()

    if username not in users["usernames"]:
        return False

    stored_password = users["usernames"][username]["password"]

    return bcrypt.checkpw(
        password.encode("utf-8"),
        stored_password.encode("utf-8")
    )


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""


if not st.session_state.logged_in:

    st.title("Multimedia Information Retrieval System")

    menu = st.sidebar.selectbox(
        "Menu",
        ["Login", "Sign Up"]
    )


    if menu == "Sign Up":

        st.header("Create Account")

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password"
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password"
        )


        if st.button("Register"):

            if username == "" or password == "":
                st.error("Please fill all fields.")

            elif password != confirm_password:
                st.error("Passwords do not match.")

            elif register_user(username, password):
                st.success(
                    "Account created successfully. Please login."
                )

            else:
                st.error(
                    "Username already exists."
                )


    else:

        st.header("Login")

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password"
        )


        if st.button("Login"):

            if verify_user(username, password):

                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:

                st.error(
                    "Invalid username or password."
                )



    st.write(
        "Welcome to the Multimedia Information Retrieval System"
    )
else:

    st.sidebar.success(
        f"Logged in as {st.session_state.username}"
    )

    if st.sidebar.button("Logout"):

        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()


    st.title("Multimedia Information Retrieval System")

    media_type = st.selectbox(
        "Select Multimedia Type",
        [
            "Image",
            "Audio",
            "Video",
            "Text"
        ]
    )

    st.divider() 
    
    top_k = st.sidebar.slider(
    "Top-K Results",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)   

    if media_type == "Image":

        uploaded_image = st.file_uploader(
            "Upload an Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_image:

            image = Image.open(uploaded_image)

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

            if st.button("Search Similar Images"):

                temp_path = "temp_image.jpg"

                image.save(temp_path)

                query_features = get_image_features(temp_path)

                similarity = cosine_similarity(
                    [query_features],
                    embeddings
                )[0]

                top_indices = similarity.argsort()[-top_k:][::-1]

                st.subheader("Similar Images")

                cols = st.columns(min(top_k, 5))

                for i, index in enumerate(top_indices):
                    
                    if i % 5==0:
                        cols = st.columns(min(5, top_k - i))


                    with cols[i % 5]:

                        result_image = filenames[index]

                        st.image(
                            result_image,
                            width=150
                        )

                        st.write(
                            f"Score: {similarity[index]*100:.2f}%"
                        )

                os.remove(temp_path)


    elif media_type == "Audio":

        uploaded_audio = st.file_uploader(
            "Upload an Audio File",
            type=["mp3", "wav", "ogg"]
        )

        if uploaded_audio:

            st.audio(uploaded_audio)

            if st.button("Search Similar Audio"):

                st.success(
                    "Audio uploaded successfully."
                )

                st.info(
                    "Audio retrieval model will be connected here."
                )


    elif media_type == "Video":

        uploaded_video = st.file_uploader(
            "Upload a Video",
            type=["mp4", "avi", "mov", "mkv"]
        )

        if uploaded_video:

            st.video(uploaded_video)

            if st.button("Search Similar Videos"):

                st.success(
                    "Video uploaded successfully."
                )

                st.info(
                    "Video retrieval model will be connected here."
                )


    elif media_type == "Text":

        st.subheader("Text Retrieval")

        query = st.text_area(
            "Enter your search query"
        )

        if st.button("Search Similar Text"):

            if query.strip() == "":

                st.warning(
                    "Please enter some text."
                )

            else:

                query_vector = vectorizer.transform(
                    [query]
                )

                similarity = cosine_similarity(
                    query_vector,
                    text_vectors
                ).flatten()

                top_indices = similarity.argsort()[-top_k:][::-1]

                st.subheader(
                    "Top Matching Results"
                )

                for index in top_indices:

                    st.write("Description:")

                    st.write(
                        text_df.iloc[index]["description"]
                    )

                    st.write("Tags:")

                    st.write(
                        text_df.iloc[index]["tags"]
                    )

                    st.write(
                        f"Similarity Score: {similarity[index]:.2f}"
                    )

                    st.divider()
                    