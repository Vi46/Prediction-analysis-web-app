from flask import Flask, render_template, request
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the pre-trained sentiment analysis model
model_path = "SaAnalysisCodeUsingRSynthetic.pkl"
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer along with its vocabulary
vectorizer_path = "TfidfVectorizerRSynthetic.pkl"
with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text):
    # Add your text preprocessing steps here
    # Example: Removing special characters, lowercasing, etc.
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']  # Assuming you have a text input field in the form

        # Preprocess the text data
        text = preprocess_text(text)

        # Vectorize the text using the loaded vectorizer
        # (ensure vectorizer settings match the training data)
        text_vector = vectorizer.transform([text])

        # Perform sentiment analysis using the loaded model
        sentiment = model.predict(text_vector)[0]

        return render_template('prediction.html', sentiment=sentiment)

    # Handle errors or other scenarios
    return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)



