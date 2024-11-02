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
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# Configuration
NUM_LANDMARKS = 42
IMG_SIZE = 224
MODEL_PATH = "path/to/your/trained/rnn_model.h5"  # **REPLACE this!**
LABEL_ENCODER_PATH = "path/to/your/label_encoder.pkl"  # **REPLACE this!**
SPEECH_DELAY = 3
CONFIDENCE_THRESHOLD = 0.8
AUDIO_FILE_LIFETIME = 2
UNKNOWN_LABEL = "unknown"
FRAME_WIDTH = 600
FRAME_HEIGHT = 500
VISUALIZATION_SIZE = 400
SEQUENCE_LENGTH = 10

# Initialize pygame mixer
pygame.init()
pygame.mixer.init(44100, -16, 2, 1024)

# Load label encoder
with open(LABEL_ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

# Initialize RNN model (replace with your actual model architecture if different)
rnn_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, NUM_LANDMARKS * 3)),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])
rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Load pre-trained RNN weights (optional, if available)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.9, min_tracking_confidence=0.9)

cap = cv2.VideoCapture(0)

# Global variables
audio_file_counter = 0
last_speech_time = 0
previous_label = ""
frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
audio_executor = ThreadPoolExecutor(max_workers=1)

def play_audio(filename):
    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        delete_timer = threading.Timer(AUDIO_FILE_LIFETIME, os.remove, args=(filename,))
        delete_timer.start()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Error playing audio: {e}")

def extract_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    return landmarks

def normalize_landmarks(landmarks):
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
    normalized_landmarks = normalize_landmarks(landmarks)
    landmarks_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    if normalized_landmarks:
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            x1, y1 = int(normalized_landmarks[start_idx][0] * (img_size - 1)), int(
                normalized_landmarks[start_idx][1] * (img_size - 1))
            x2, y2 = int(normalized_landmarks[end_idx][0] * (img_size - 1)), int(
                normalized_landmarks[end_idx][1] * (img_size - 1))
            cv2.line(landmarks_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

        for lm in normalized_landmarks:
            x, y = int(lm[0] * (img_size - 1)), int(lm[1] * (img_size - 1))
            cv2.circle(landmarks_image, (x, y), 5, (255, 0, 0), 3)  # -1 filled

    return landmarks_image, np.expand_dims(landmarks_image / 255.0, axis=0)

def smooth_predictions(new_prediction):
    prediction_history.append(new_prediction)
    return np.mean(prediction_history, axis=0)

# Visualization window
cv2.namedWindow("Preprocessed Landmarks", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Preprocessed Landmarks", VISUALIZATION_SIZE, VISUALIZATION_SIZE)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    predicted_label = UNKNOWN_LABEL
    confidence = 0.0
    preprocessed_viz = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            landmarks = extract_landmarks(hand_landmarks)

            for lm in landmarks:  # Draw landmarks on the original frame
                x = int(lm[0] * frame.shape[1])
                y = int(lm[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)


            preprocessed_viz, _ = preprocess_landmarks(landmarks)
            normalized_landmarks = normalize_landmarks(landmarks)
            frame_sequence.append(np.array(normalized_landmarks).flatten())


            if len(frame_sequence) == SEQUENCE_LENGTH:
                input_sequence = np.expand_dims(np.array(frame_sequence), axis=0)

                raw_predictions = rnn_model.predict(input_sequence, verbose=0)[0]
                filtered_predictions = smooth_predictions(raw_predictions)
                predicted_class_index = np.argmax(filtered_predictions)

                try:
                    predicted_label = le.inverse_transform([predicted_class_index])[0]
                except IndexError:
                    predicted_label = UNKNOWN_LABEL

                confidence = filtered_predictions[predicted_class_index]

                hand_label_position = (int(landmarks[0][0] * frame.shape[1]), int(landmarks[0][1] * frame.shape[0]) - 30)
                cv2.putText(frame, f"{predicted_label} ({confidence * 100:.2f}%)",
                            hand_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    else:
        cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Clear the frame sequence buffer when no hands are detected
        frame_sequence.clear()


    cv2.imshow('Real-time ASL Detection', frame)
    cv2.imshow("Preprocessed Landmarks", preprocessed_viz)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
audio_executor.shutdown()