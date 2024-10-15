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

# Updated Configuration
NUM_LANDMARKS = 42
IMG_SIZE = 128  # Changed from 320 to 128
SEQUENCE_LENGTH = 20  # Changed from 30 to 20
MODEL_PATH = "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_FILES/models/gesture_model.h5"
LABEL_ENCODER_PATH_CNN = "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_FILES/data/labels/le_cnn.pkl"
LABEL_ENCODER_PATH_SEQ = "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_FILES/data/labels/le_seq.pkl"
SPEECH_DELAY = 3
CONFIDENCE_THRESHOLD = 0.8
AUDIO_FILE_LIFETIME = 2
FRAME_WIDTH = 600
FRAME_HEIGHT = 500

# Initialize pygame mixer for audio
pygame.init()
pygame.mixer.init()

# Load trained model and label encoders
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_ENCODER_PATH_CNN, 'rb') as f:
    le_cnn = pickle.load(f)
with open(LABEL_ENCODER_PATH_SEQ, 'rb') as f:
    le_seq = pickle.load(f)

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

# Initialize deques to store frame sequences
cnn_frame_sequence = deque(maxlen=1)  # Only need the latest frame for CNN
sequence_frame_sequence = deque(maxlen=SEQUENCE_LENGTH)

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
            cnn_frame_sequence.append(preprocessed_image)
            sequence_frame_sequence.append(preprocessed_image)

            # Draw landmarks on the frame
            for lm in landmarks:
                x = int(lm[0] * frame.shape[1])
                y = int(lm[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # If we have collected enough frames, make a prediction
    if len(cnn_frame_sequence) == 1 and len(sequence_frame_sequence) == SEQUENCE_LENGTH:
        cnn_input = np.expand_dims(np.array(cnn_frame_sequence), axis=0)
        sequence_input = np.expand_dims(np.array(sequence_frame_sequence), axis=0)
        
        cnn_predictions, sequence_predictions = model.predict([cnn_input, sequence_input])
        
        cnn_predicted_class_index = np.argmax(cnn_predictions[0])
        sequence_predicted_class_index = np.argmax(sequence_predictions[0])
        
        cnn_confidence = cnn_predictions[0][cnn_predicted_class_index]
        sequence_confidence = sequence_predictions[0][sequence_predicted_class_index]

        try:
            cnn_predicted_label = le_cnn.inverse_transform([cnn_predicted_class_index])[0]
            sequence_predicted_label = le_seq.inverse_transform([sequence_predicted_class_index])[0]
        except IndexError:
            cnn_predicted_label = "unknown"
            sequence_predicted_label = "unknown"

        # Display the predictions
        cv2.putText(frame, f"CNN: {cnn_predicted_label} ({cnn_confidence * 100:.2f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Sequence: {sequence_predicted_label} ({sequence_confidence * 100:.2f}%)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Text-to-speech and audio playback (using sequence prediction)
        current_time = time.time()
        if (sequence_predicted_label != previous_label and 
            (current_time - last_speech_time) >= SPEECH_DELAY and 
            sequence_confidence >= CONFIDENCE_THRESHOLD):
            
            tts = gTTS(text=sequence_predicted_label, lang='en')
            audio_filename = f"temp_{audio_file_counter}.mp3"
            tts.save(audio_filename)
            audio_file_counter += 1

            audio_executor.submit(play_audio, audio_filename)

            last_speech_time = current_time
            previous_label = sequence_predicted_label

    cv2.imshow("Real-time ASL Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
audio_executor.shutdown()