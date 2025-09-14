from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
from nb_model import mulitnomialNB

app = Flask(__name__)
CORS(app)


vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
