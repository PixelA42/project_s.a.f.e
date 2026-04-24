from flask import Flask, request, jsonify
from flask_ import CORS
import random

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return {"message": "Backend running"}

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('file')

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # ---- MOCK LOGIC (replace with ML later) ----
    labels = ["safe", "prank", "high_risk"]
    label = random.choice(labels)

    response = {
        "label": label,
        "final_score": random.randint(60, 95),
        "spectral_score": random.randint(50, 90),
        "intent_score": random.randint(50, 95)
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)