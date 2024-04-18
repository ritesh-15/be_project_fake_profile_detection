from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
import numpy as np
import joblib

from sklearn.decomposition import PCA

from nltk.corpus import stopwords
import string
from nltk.stem import SnowballStemmer
import nltk
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd

tfidf = joblib.load("tfidf.joblib")

nltk.download("stopwords")
nltk.download("punkt")

app = Flask(__name__, static_url_path="/static")
model = load_model("model.h5")

# encoders
name_encoder = joblib.load("encoders/name_encoder.joblib")
screen_name_encoder = joblib.load("encoders/screen_name_encoder.joblib")
location_encoder = joblib.load("encoders/location_encoder.joblib")
lang_encoder = joblib.load("encoders/lang_encoder.joblib")
profile_background_color_encoder = joblib.load(
    "encoders/profile_background_color_encoder.joblib"
)
profile_link_color_encoder = joblib.load("encoders/profile_link_color_encoder.joblib")
profile_sidebar_fill_color_encoder = joblib.load(
    "encoders/profile_sidebar_fill_color_encoder.joblib"
)
sidebar_border_color_encoder = joblib.load(
    "encoders/profile_sidebar_border_color_encoder.joblib"
)
profile_text_color_encoder = joblib.load("encoders/profile_text_color_encoder.joblib")


def pca(x):
    pca = joblib.load("pca.joblib")
    std = joblib.load("std.joblib")
    pcaData = pca.transform(x)
    pca_df = pd.DataFrame(pcaData)
    pca_df = std.transform(pca_df)
    return pca_df


def transform_text(text, lang):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if lang == "en":
            if i not in stopwords.words("english") and i not in string.punctuation:
                y.append(i)
        elif lang == "fr":
            if i not in stopwords.words("french") and i not in string.punctuation:
                y.append(i)
        elif lang == "sn":
            if i not in stopwords.words("spanish") and i not in string.punctuation:
                y.append(i)
        elif lang == "de":
            if i not in stopwords.words("german") and i not in string.punctuation:
                y.append(i)
        elif lang == "nl":
            if i not in stopwords.words("dutch") and i not in string.punctuation:
                y.append(i)
        elif lang == "tr":
            if i not in stopwords.words("turkish") and i not in string.punctuation:
                y.append(i)
        elif lang == "it":
            if i not in stopwords.words("italian") and i not in string.punctuation:
                y.append(i)

    text = y[:]
    y.clear()

    if lang == "gl" or lang == "tr":
        return ""

    language = "english"

    if lang == "en":
        language = "english"
    elif lang == "sn":
        language = "spanish"
    elif lang == "it":
        language = "italian"
    elif lang == "fr":
        language = "french"
    elif lang == "de":
        language = "german"
    elif lang == "nl":
        language = "dutch"
    elif lang == "gl":
        language = "greenlandic"
    elif lang == "tr":
        language = "turkish"

    ps = SnowballStemmer(language)

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


def preprocessing(data):
    preprocessed_data = data

    # description
    transformed_description = transform_text(data["description"], data["language"])
    tfidf_description = tfidf.transform([transformed_description]).toarray()

    for feature, value in data.items():
        if value in name_encoder.classes_:
            preprocessed_data[feature] = name_encoder.transform([data[feature]])[0]
        elif value in screen_name_encoder.classes_:
            preprocessed_data[feature] = screen_name_encoder.transform([data[feature]])[
                0
            ]
        elif value in location_encoder.classes_:
            preprocessed_data[feature] = location_encoder.transform([data[feature]])[0]
        elif value in lang_encoder.classes_:
            preprocessed_data[feature] = lang_encoder.transform([data[feature]])[0]
        elif value in profile_link_color_encoder.classes_:
            preprocessed_data[feature] = profile_link_color_encoder.transform(
                [data[feature]]
            )[0]
        elif value in profile_background_color_encoder.classes_:
            preprocessed_data[feature] = profile_background_color_encoder.transform(
                [data[feature]]
            )[0]
        elif feature in profile_text_color_encoder.classes_:
            preprocessed_data[feature] = profile_text_color_encoder.transform(
                [data[feature]]
            )[0]
        elif feature in profile_sidebar_fill_color_encoder.classes_:
            preprocessed_data[feature] = profile_sidebar_fill_color_encoder.transform(
                [data[feature]]
            )[0]
        elif feature in sidebar_border_color_encoder.classes_:
            preprocessed_data[feature] = sidebar_border_color_encoder.transform(
                [data[feature]]
            )[0]
        elif type(value) == str and feature != "description":

            preprocessed_data[feature] = -1

    return (preprocessed_data, tfidf_description)


@app.route("/predict", methods=["POST"])
def predict():
    name = request.form.get("name")
    screen_name = request.form.get("screenName")
    description = request.form.get("description")
    statuses_count = int(request.form.get("statusesCount"))
    followers_count = int(request.form.get("followersCount"))
    friends_count = int(request.form.get("friendsCount"))
    favorites_count = int(request.form.get("favoritesCount"))
    listed_count = int(request.form.get("listedCount"))
    language = request.form.get("language")
    location = request.form.get("location")
    text_color = request.form.get("profileTextColor")
    sidebar_border_color = request.form.get("sidebarBorderColor")
    sidebar_fill_color = request.form.get("sidebarFillColor")
    background_color = request.form.get("backgroundColor")
    profile_link_color = request.form.get("profileLinkColor")

    data = {
        "name": name,
        "screen_name": screen_name,
        "description": description,
        "statuses_count": statuses_count,
        "followers_count": followers_count,
        "friends_count": friends_count,
        "favorites_count": favorites_count,
        "listed_count": listed_count,
        "language": language,
        "location": location,
        "default_profile": 1,
        "default_profile_image": 0,
        "geo_enabled": 1,
        "profile_use_background_image": 0,
        "profile_link_color": profile_link_color,
        "background_color": background_color,
        "sidebar_fill_color": sidebar_fill_color,
        "sidebar_border_color": sidebar_border_color,
        "text_color": text_color,
        "protected": 0,
        "verified": 0,
        "year": 0,
        "month": 0,
        "day": 0,
        "hour": 0,
        "utc_offset": 0,
        "id": 616597381,
        "profile_background_tile": 0,
    }

    preprocessed_data, tfid = preprocessing(data)

    del preprocessed_data["description"]

    x = np.array(list(preprocessed_data.values())).reshape(1, -1)
    x = np.concatenate((x, tfid), axis=1)

    data_reduction = pca(x)

    print(data_reduction.shape)

    model.layers[0].input_shape = (data_reduction.shape[1],)
    prediction = model.predict(data_reduction)[0][0]
    is_fake = True if prediction > 0.5 else False

    if is_fake:
        return redirect("/fake")

    return redirect("/valid")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/fake")
def fake():
    return render_template("fake.html")


@app.route("/valid")
def valid():
    return render_template("notfake.html")


if __name__ == "__main__":
    app.run(debug=True)
