import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
import pygame
import time
import os
from gtts import gTTS
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

# Configuration
NUM_LANDMARKS = 42
IMG_SIZE = 320
SEQUENCE_LENGTH = 30
MODELS_CONFIG = {
    'cnn': {
        'path': "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_FILES/models/gesture_model_cnn.h5",
        'weight': 0.6  # Confidence weight for CNN model
    },
    'rnn': {
        'path': "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_FILES/models/gesture_model_rnn.h5",
        'weight': 0.4  # Confidence weight for RNN model
    }
}
LABEL_ENCODER_PATH = "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_FILES/data/labels/gesture_label_encoder.pkl"
SPEECH_DELAY = 3
CONFIDENCE_THRESHOLD = 0.8
AUDIO_FILE_LIFETIME = 2
FRAME_WIDTH = 600
FRAME_HEIGHT = 500

class ASLDetector:
    def __init__(self):
        # Initialize pygame mixer for audio
        pygame.init()
        pygame.mixer.init()

        # Load models and label encoder
        self.models = self._load_models()
        self.le = self._load_label_encoder()

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)

        # Initialize other variables
        self.audio_file_counter = 0
        self.last_speech_time = 0
        self.previous_label = ""
        self.frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.audio_executor = ThreadPoolExecutor(max_workers=2)

    def _load_models(self) -> Dict[str, tf.keras.Model]:
        """Load all models specified in MODELS_CONFIG."""
        models = {}
        for model_name, config in MODELS_CONFIG.items():
            try:
                models[model_name] = tf.keras.models.load_model(config['path'])
                print(f"Successfully loaded {model_name} model")
            except Exception as e:
                print(f"Error loading {model_name} model: {e}")
                models[model_name] = None
        return models

    def _load_label_encoder(self):
        """Load the label encoder."""
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            return pickle.load(f)

    def extract_landmarks(self, hand_landmarks) -> List[List[float]]:
        """Extract landmarks from MediaPipe hand detection."""
        return [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

    def normalize_landmarks(self, landmarks: List[List[float]]) -> List[List[float]]:
        """Normalize landmark coordinates."""
        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        return [
            [(x - min_x) / (max_x - min_x), (y - min_y) / (max_y - min_y), z]
            for x, y, z in landmarks
        ]

    def preprocess_landmarks(self, landmarks: List[List[float]], img_size: int = IMG_SIZE) -> np.ndarray:
        """Convert landmarks to image representation."""
        normalized_landmarks = self.normalize_landmarks(landmarks)
        landmarks_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        if normalized_landmarks:
            for connection in self.mp_hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                x1, y1 = int(normalized_landmarks[start_idx][0] * (img_size - 1)), int(normalized_landmarks[start_idx][1] * (img_size - 1))
                x2, y2 = int(normalized_landmarks[end_idx][0] * (img_size - 1)), int(normalized_landmarks[end_idx][1] * (img_size - 1))
                cv2.line(landmarks_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

            for lm in normalized_landmarks:
                x, y = int(lm[0] * (img_size - 1)), int(lm[1] * (img_size - 1))
                cv2.circle(landmarks_image, (x, y), 5, (255, 0, 0), 3)

        return landmarks_image

    def get_ensemble_prediction(self, input_data: np.ndarray) -> Tuple[str, float]:
        """Combine predictions from multiple models."""
        predictions = {}
        weights = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            if model is not None:
                pred = model.predict(input_data, verbose=0)[0]
                predictions[model_name] = pred
                weights[model_name] = MODELS_CONFIG[model_name]['weight']

        if not predictions:
            return "unknown", 0.0

        # Combine predictions using weighted average
        combined_pred = np.zeros_like(predictions[list(predictions.keys())[0]])
        total_weight = sum(weights.values())
        
        for model_name, pred in predictions.items():
            weight = weights[model_name] / total_weight
            combined_pred += pred * weight

        predicted_class_index = np.argmax(combined_pred)
        confidence = combined_pred[predicted_class_index]

        try:
            predicted_label = self.le.inverse_transform([predicted_class_index])[0]
        except IndexError:
            predicted_label = "unknown"

        return predicted_label, confidence

    def play_audio(self, filename: str):
        """Play and clean up audio file."""
        try:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            os.remove(filename)
        except Exception as e:
            print(f"Error playing audio: {e}")

    def run(self):
        """Main detection loop."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = self.extract_landmarks(hand_landmarks)
                    preprocessed_image = self.preprocess_landmarks(landmarks)
                    self.frame_sequence.append(preprocessed_image)

                    # Draw landmarks on frame
                    for lm in landmarks:
                        x = int(lm[0] * frame.shape[1])
                        y = int(lm[1] * frame.shape[0])
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Make prediction if we have enough frames
            if len(self.frame_sequence) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(np.array(self.frame_sequence), axis=0)
                predicted_label, confidence = self.get_ensemble_prediction(input_data)

                # Display predictions
                cv2.putText(frame, f"Prediction: {predicted_label} ({confidence * 100:.2f}%)",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Handle text-to-speech
                current_time = time.time()
                if (predicted_label != self.previous_label and 
                    (current_time - self.last_speech_time) >= SPEECH_DELAY and 
                    confidence >= CONFIDENCE_THRESHOLD):
                    
                    tts = gTTS(text=predicted_label, lang='en')
                    audio_filename = f"temp_{self.audio_file_counter}.mp3"
                    tts.save(audio_filename)
                    self.audio_file_counter += 1

                    self.audio_executor.submit(self.play_audio, audio_filename)

                    self.last_speech_time = current_time
                    self.previous_label = predicted_label

            cv2.imshow("Real-time ASL Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.audio_executor.shutdown()

if __name__ == "__main__":
    detector = ASLDetector()
    detector.run()