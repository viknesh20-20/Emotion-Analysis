from flask import Flask, render_template, request, jsonify
import pickle
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and tokenizer
model = load_model('emotion_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)


# Route for rendering the HTML page
@app.route('/')
def index():
    return render_template('index.html')


# Route for emotion prediction
@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        data = request.get_json()  # Get the JSON data from the frontend
        text = data['text']  # Extract the input text

        print(f"Received text: {text}")  # Debugging print statement

        # Tokenize the input text and make prediction
        sequences = tokenizer.texts_to_sequences([text])
        padded_sequences = np.pad(sequences, [(0, 0), (0, 50 - len(sequences[0]))], mode='constant')

        # Predict emotion
        prediction = model.predict(padded_sequences)
        predicted_emotion = np.argmax(prediction, axis=1)

        emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
                    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
                    'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
                    'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
                    'remorse', 'sadness', 'surprise', 'neutral']
        predicted_emotions = [emotions[i] for i in predicted_emotion]

        print(f"Predicted emotions: {predicted_emotions}")  # Debugging print statement

        # Return the predicted emotion
        return jsonify({'predicted_emotions': predicted_emotions})

    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Error logging
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
