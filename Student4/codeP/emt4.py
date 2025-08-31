import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class EmotionDetector:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = RandomForestClassifier()
        self.emotions = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        self.features = []
        self.labels = []

    def extract_features(self, file_path):
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled

    def load_data(self):
        print("Loading dataset and extracting features...")
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    path = os.path.join(root, file)
                    try:
                        emotion_code = file.split("-")[2]
                        emotion_label = self.emotions.get(emotion_code)
                        if emotion_label:
                            features = self.extract_features(path)
                            self.features.append(features)
                            self.labels.append(emotion_label)
                    except Exception as e:
                        print(f"Error with file {file}: {e}")

    def train_model(self):
        print("Training model...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy:.2f}")
        return accuracy

    def predict_emotion(self, file_path):
        print(f"Predicting emotion for: {file_path}")
        try:
            features = self.extract_features(file_path).reshape(1, -1)
            prediction = self.model.predict(features)[0]
            return prediction
        except Exception as e:
            print("Error in prediction:", e)
            return None
