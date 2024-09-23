import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import threading
import pygame
import time
import os
from gtts import gTTS

# Initialize pygame
pygame.mixer.init()

# --------------------- Configuration ---------------------
NUM_LANDMARKS = 42
IMG_SIZE = 227
MODEL_PATH = "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_FILES/models/asl_model.h5"  # Update with your model path
LABEL_ENCODER_PATH = "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_FILES/data/labels/label_encoder.pkl"  # Update with your label encoder path
SPEECH_DELAY = 3
CONFIDENCE_THRESHOLD = 0.8
AUDIO_FILE_LIFETIME = 3
DEBUG_FRAME_LIFETIME = 5
UNKNOWN_LABEL = "unknown"
# ----------------------------------------------------------

# Load trained model and label encoder
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

# Global variables
audio_file_counter = 0
last_speech_time = 0
previous_label = ""
debug_frame_time = 0

def play_audio(filename):
    """Plays an audio file asynchronously and schedules it for deletion."""
    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        delete_timer = threading.Timer(AUDIO_FILE_LIFETIME, os.remove, args=(filename,))
        delete_timer.start()
    except Exception as e:
        print(f"Error playing audio: {e}")

def extract_landmarks(hand_landmarks):
    """Extracts hand landmarks from MediaPipe results."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    return landmarks

def normalize_landmarks(landmarks):
    """Normalize landmarks to be invariant to scale and translation."""
    x_coords = [lm[0] for lm in landmarks]
    y_coords = [lm[1] for lm in landmarks]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    normalized_landmarks = [
        [(x - min_x) / (max_x - min_x), (y - min_y) / (max_y - min_y), z]
        for x, y, z in landmarks
    ]
    return normalized_landmarks

def preprocess_landmarks(landmarks, img_size=IMG_SIZE):
    """Preprocess landmarks to create an input image compatible with the model."""
    normalized_landmarks = normalize_landmarks(landmarks)

    landmarks_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    for lm in normalized_landmarks:
        x, y = int(lm[0] * (img_size - 1)), int(lm[1] * (img_size - 1))
        cv2.circle(landmarks_image, (x, y), 10, (255, 0, 0), 3)

    landmarks_image = landmarks_image / 255.0
    return np.expand_dims(landmarks_image, axis=0)

class MovingAverageFilter:
    """Applies a moving average filter to smooth predictions."""
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.predictions = []

    def update(self, new_prediction):
        self.predictions.append(new_prediction)
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
        return np.mean(self.predictions, axis=0)

ma_filter = MovingAverageFilter()

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    predicted_label = UNKNOWN_LABEL
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = extract_landmarks(hand_landmarks)
            for lm in landmarks:
                x = int(lm[0] * frame.shape[1])
                y = int(lm[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            input_data = preprocess_landmarks(landmarks)

            # Make prediction
            raw_predictions = model.predict(input_data)[0]
            filtered_predictions = ma_filter.update(raw_predictions)
            predicted_class_index = np.argmax(filtered_predictions)

            # Get predicted label, handling potential IndexError
            try:
                predicted_label = le.inverse_transform([predicted_class_index])[0]
            except IndexError:
                predicted_label = UNKNOWN_LABEL

            confidence = filtered_predictions[predicted_class_index]

            cv2.putText(frame, f"Prediction: {predicted_label} ({confidence * 100:.2f}%)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            normalized_image = (input_data[0] * 255).astype(np.uint8)
            cv2.imshow("Normalized Hand", normalized_image)

    else:  # No hand detected
        cv2.putText(frame, f"Prediction: {predicted_label} ",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Text-to-Speech 
    current_time = time.time()
    if (predicted_label != previous_label and
        (current_time - last_speech_time) >= SPEECH_DELAY and
        confidence >= CONFIDENCE_THRESHOLD and
        predicted_label != UNKNOWN_LABEL):

        tts = gTTS(text=predicted_label, lang='en')
        audio_filename = f"temp_{audio_file_counter}.mp3"
        audio_temp_folder = "C:/Users/ronde/PROJECTS/ASL_TO_TEXT/temp_files"  # Make sure this folder exists
        save_path = os.path.join(audio_temp_folder, audio_filename)
        tts.save(save_path)
        audio_file_counter += 1

        audio_thread = threading.Thread(target=play_audio, args=(save_path,))
        audio_thread.daemon = True
        audio_thread.start()

        last_speech_time = current_time
        previous_label = predicted_label

    # Debug frame handling 
    if debug_frame_time and (time.time() - debug_frame_time) >= DEBUG_FRAME_LIFETIME:
        frame = np.zeros_like(frame)
        debug_frame_time = 0

    cv2.imshow("Real-time ASL Detection", frame)

    if cv2.getWindowProperty("Real-time ASL Detection", cv2.WND_PROP_VISIBLE) >= 1:
        debug_frame_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()