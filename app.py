import re
import joblib
import nltk
from flask import Flask, render_template, request, jsonify

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ── Load saved model & vectorizer ──────────────────────────────────────────
model     = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# ── Text cleaning (must match notebook preprocessing) ──────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Please enter some text.'}), 400

    cleaned   = clean_text(text)
    vec       = vectorizer.transform([cleaned])
    sentiment = model.predict(vec)[0]

    # Confidence scores (probability) — LinearSVC uses decision_function
    try:
        proba  = model.predict_proba(vec)[0]
        labels = model.classes_.tolist()
        scores = {label: round(float(p) * 100, 1) for label, p in zip(labels, proba)}
    except AttributeError:
        # LinearSVC fallback
        df_vals = model.decision_function(vec)[0]
        labels  = model.classes_.tolist()
        # Softmax-like normalisation
        exp_v   = [2.718 ** v for v in df_vals]
        total   = sum(exp_v)
        scores  = {label: round(e / total * 100, 1) for label, e in zip(labels, exp_v)}

    return jsonify({
        'sentiment': sentiment,
        'scores':    scores,
        'cleaned':   cleaned
    })


if __name__ == '__main__':
    app.run(debug=True)
