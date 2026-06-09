from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("dt_model.pkl", "rb"))
vectorizer = pickle.load(open("vec_model.pkl", "rb"))

def preprocess_bangla(text):
    text = str(text)

    text = re.sub(r"http\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    return text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    text = request.form["comment"]

    clean_text = preprocess_bangla(text)

    vec_data = vectorizer.transform([clean_text])

    prediction = model.predict(vec_data)[0]

    labels = {
        0: "Non-Extremist",
        1: "Extremist"
    }

    result = labels[prediction]

    return render_template(
        "index.html",
        prediction_text=result
    )

if __name__ == "__main__":
    app.run(debug=True)