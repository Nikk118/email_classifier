from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  

class mulitnomialNB():
    def fit(self, x, y):
        self.classes = np.unique(y)
        self.class_log_prior = {}
        self.feature_log_prior = {}
        self.alpha = 1
        total_docs = len(y)
        for c in self.classes:
            x_c = x[y == c]
            total_c_docs = x_c.shape[0]
            self.class_log_prior[c] = np.log(total_c_docs / total_docs)
            words_count = x_c.sum(axis=0) + self.alpha
            total_words = words_count.sum()
            self.feature_log_prior[c] = np.log(words_count / total_words)

    def predict(self, x):
        results = []
        for i in range(x.shape[0]):
            sample = x[i]
            class_score = {}
            for c in self.classes:
                log_prior = self.class_log_prior[c]
                log_liklihood = sample @ self.feature_log_prior[c].T
                class_score[c] = log_prior + log_liklihood
            predict_class = max(class_score, key=class_score.get)
            results.append(predict_class)
        return np.array(results)


vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")

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
