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
IMG_SIZE = 224
SEQUENCE_LENGTH = 30
MODELS_CONFIG = {
    'cnn': {
        'path': "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_FILES/models/asl_mobilenet_model(1).h5",
        'label_encoder': "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_FILES/data/labels/label_encoder.pkl",
        'weight': 0.7  # Confidence weight for CNN model
    },
    'rnn': {
        'path': "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_FILES/models/rnn_model.keras",
        'label_encoder': "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_FILES/data/labels/rnn_label_encoder.pkl",
        'weight': 0.3  # Confidence weight for RNN model
    }
}

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

        # Load models and label encoders
        self.models, self.label_encoders = self._load_models_and_encoders()

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

    def _load_models_and_encoders(self) -> Tuple[Dict[str, tf.keras.Model], Dict[str, object]]:
        """Load all models and their corresponding label encoders."""
        models = {}
        label_encoders = {}
        
        for model_name, config in MODELS_CONFIG.items():
            try:
                if os.path.exists(config['path']):
                    models[model_name] = tf.keras.models.load_model(config['path'])
                    print(f"Model {model_name} input shape:", models[model_name].input_shape)
                else:
                    print(f"Model file not found for {model_name}")

                if os.path.exists(config['label_encoder']):
                    with open(config['label_encoder'], 'rb') as f:
                        label_encoders[model_name] = pickle.load(f)
                    print(f"Successfully loaded {model_name} model and label encoder")
                else:
                    print(f"Label encoder file not found for {model_name}")
                    
            except Exception as e:
                print(f"Error loading {model_name} model or label encoder: {e}")
                models[model_name] = None
                label_encoders[model_name] = None
        
        return models, label_encoders

    def preprocess_for_cnn(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for CNN model."""
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def preprocess_for_rnn(self, sequence: np.ndarray) -> np.ndarray:
        """Preprocess sequence for RNN model."""
        sequence = np.array(sequence)
        sequence = sequence.astype('float32') / 255.0
        return np.expand_dims(sequence, axis=0)

    def get_ensemble_prediction(self, frame: np.ndarray, sequence: np.ndarray) -> Tuple[str, float]:
        """Combine predictions from multiple models using their respective label encoders."""
        predictions = {}
        weights = {}
        
        # CNN Prediction
        if self.models.get('cnn') is not None:
            try:
                cnn_input = self.preprocess_for_cnn(frame)
                pred_cnn = self.models['cnn'].predict(cnn_input, verbose=0)[0]
                predictions['cnn'] = pred_cnn
                weights['cnn'] = MODELS_CONFIG['cnn']['weight']
            except Exception as e:
                print(f"Error in CNN prediction: {e}")

        # RNN Prediction
        if self.models.get('rnn') is not None:
            try:
                rnn_input = self.preprocess_for_rnn(sequence)
                pred_rnn = self.models['rnn'].predict(rnn_input, verbose=0)[0]
                predictions['rnn'] = pred_rnn
                weights['rnn'] = MODELS_CONFIG['rnn']['weight']
            except Exception as e:
                print(f"Error in RNN prediction: {e}")

        if not predictions:
            return "unknown", 0.0

        # Process predictions
        final_predictions = {}
        for model_name, pred in predictions.items():
            if self.label_encoders.get(model_name) is not None:
                predicted_class_index = np.argmax(pred)
                try:
                    predicted_label = self.label_encoders[model_name].inverse_transform([predicted_class_index])[0]
                    confidence = pred[predicted_class_index]
                    final_predictions[predicted_label] = final_predictions.get(predicted_label, 0) + (confidence * weights[model_name])
                except Exception as e:
                    print(f"Error processing prediction for {model_name}: {e}")

        if final_predictions:
            predicted_label = max(final_predictions.items(), key=lambda x: x[1])
            return predicted_label[0], predicted_label[1] / sum(weights.values())
        
        return "unknown", 0.0

    def extract_landmarks(self, hand_landmarks) -> List[float]:
        """Extract landmarks from hand landmarks."""
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks

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

            current_frame_preprocessed = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = self.extract_landmarks(hand_landmarks)
                    self.frame_sequence.append(landmarks)
                    current_frame_preprocessed = frame

                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )

            if current_frame_preprocessed is not None:
                predicted_label, confidence = self.get_ensemble_prediction(
                    current_frame_preprocessed,
                    list(self.frame_sequence)
                )

                cv2.putText(frame, f"Prediction: {predicted_label} ({confidence * 100:.2f}%)",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                current_time = time.time()
                if (predicted_label != self.previous_label and 
                    predicted_label != "unknown" and
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

    def play_audio(self, filename: str):
        """Play an audio file asynchronously."""
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        time.sleep(AUDIO_FILE_LIFETIME)
        pygame.mixer.music.stop()
        os.remove(filename)

    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.audio_executor.shutdown()
        pygame.quit()

if __name__ == "__main__":
    detector = ASLDetector()
    detector.run()
