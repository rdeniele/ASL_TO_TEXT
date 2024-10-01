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

# Configuration
NUM_LANDMARKS = 42
IMG_SIZE = 320
SEQUENCE_LENGTH = 30
MODEL_PATH = "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_FILES/models/gesture_model.h5"
LABEL_ENCODER_PATH = "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_FILES/data/labels/gesture_label_encoder.pkl"
SPEECH_DELAY = 3
CONFIDENCE_THRESHOLD = 0.8
AUDIO_FILE_LIFETIME = 2
FRAME_WIDTH = 600
FRAME_HEIGHT = 500

# Initialize pygame mixer for audio
pygame.init()
pygame.mixer.init()

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

def play_audio(filename):
    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        os.remove(filename)
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

    if normalized_landmarks:  # Check if landmarks are detected
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            x1, y1 = int(normalized_landmarks[start_idx][0] * (img_size - 1)), int(normalized_landmarks[start_idx][1] * (img_size - 1))
            x2, y2 = int(normalized_landmarks[end_idx][0] * (img_size - 1)), int(normalized_landmarks[end_idx][1] * (img_size - 1))
            cv2.line(landmarks_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    for lm in normalized_landmarks:
        x, y = int(lm[0] * (img_size - 1)), int(lm[1] * (img_size - 1))
        cv2.circle(landmarks_image, (x, y), 5, (255, 0, 0), 3)

    return landmarks_image

# Thread pool for audio playback
audio_executor = ThreadPoolExecutor(max_workers=2)

# Initialize deque to store frame sequence
frame_sequence = deque(maxlen=SEQUENCE_LENGTH)

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = extract_landmarks(hand_landmarks)
            preprocessed_image = preprocess_landmarks(landmarks)
            frame_sequence.append(preprocessed_image)

            # Draw landmarks on the frame
            for lm in landmarks:
                x = int(lm[0] * frame.shape[1])
                y = int(lm[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # If we have collected enough frames, make a prediction
    if len(frame_sequence) == SEQUENCE_LENGTH:
        input_data = np.expand_dims(np.array(frame_sequence), axis=0)
        predictions = model.predict(input_data)[0]
        predicted_class_index = np.argmax(predictions)
        confidence = predictions[predicted_class_index]

        try:
            predicted_label = le.inverse_transform([predicted_class_index])[0]
        except IndexError:
            predicted_label = "unknown"

        # Display the prediction
        cv2.putText(frame, f"Prediction: {predicted_label} ({confidence * 100:.2f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Text-to-speech and audio playback
        current_time = time.time()
        if (predicted_label != previous_label and 
            (current_time - last_speech_time) >= SPEECH_DELAY and 
            confidence >= CONFIDENCE_THRESHOLD):
            
            tts = gTTS(text=predicted_label, lang='en')
            audio_filename = f"temp_{audio_file_counter}.mp3"
            tts.save(audio_filename)
            audio_file_counter += 1

            audio_executor.submit(play_audio, audio_filename)

            last_speech_time = current_time
            previous_label = predicted_label

    cv2.imshow("Real-time ASL Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
audio_executor.shutdown()