import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
import winsound
import time
from collections import deque

# Load dlib face detector and landmarks model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Make sure this file is in the same directory

# EAR/MAR calculation functions
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[13] - mouth[19])
    B = np.linalg.norm(mouth[0] - mouth[6])
    return A / B

# Facial fatigue check
def detect_facial_fatigue(frame, ear_buffer, mar_buffer):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if not faces:
        ear_buffer.clear()
        mar_buffer.clear()
        return 0, None

    for face in faces:
        landmarks = predictor(gray, face)
        eye_left = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        eye_right = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])

        ear = (eye_aspect_ratio(eye_left) + eye_aspect_ratio(eye_right)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        ear_buffer.append(ear)
        mar_buffer.append(mar)

        if len(ear_buffer) > 5:
            ear_buffer.popleft()
            mar_buffer.popleft()

        avg_ear = np.mean(ear_buffer)
        avg_mar = np.mean(mar_buffer)
        drowsy = 1 if (avg_ear < 0.22 or avg_mar > 0.8) and len(ear_buffer) >= 5 else 0

        return drowsy, (face.left(), face.top(), face.right(), face.bottom())

    return 0, None

# Physiological fatigue simulation
def detect_phys_fatigue(bpm_buffer, eeg_buffer, phys_state_buffer):
    rr_intervals = [60000 / bpm for bpm in bpm_buffer]
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2)) if len(rr_intervals) > 1 else 100
    avg_eeg = np.mean(eeg_buffer)

    fatigued = 1 if rmssd < 15 or avg_eeg > 14 else 0
    phys_state_buffer.append(fatigued)
    if len(phys_state_buffer) > 5:
        phys_state_buffer.popleft()

    return 1 if sum(phys_state_buffer) >= 3 and len(phys_state_buffer) >= 5 else 0, rmssd, avg_eeg

# Display result
def declare_fatigue_result(state, frame, face_bbox, avg_ear, avg_mar, rmssd, avg_eeg, fscore):
    facial, phys = state
    if facial == 0 and phys == 0:
        result = "Alert"
        color = (0, 255, 0)
        winsound.Beep(500, 200)
    elif facial == 0 and phys == 1:
        result = "Fatigued"
        color = (0, 255, 255)
        winsound.Beep(700, 300)
    elif facial == 1 and phys == 0:
        result = "Drowsy"
        color = (0, 165, 255)
        winsound.Beep(900, 400)
    else:
        result = "CRITICAL"
        color = (0, 0, 255)
        winsound.Beep(1200, 500)
        print("\n*** CRITICAL FATIGUE ***")

    if face_bbox:
        x1, y1, x2, y2 = face_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Status: {result}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"F-Score: {fscore:.2f}", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    print(f"Driver Status: {result}, F-Score: {fscore:.2f}, RMSSD: {rmssd:.2f}, EEG: {avg_eeg:.2f}")
    return frame

# Main function
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        return

    ear_buffer = deque(maxlen=5)
    mar_buffer = deque(maxlen=5)
    bpm_buffer = deque([70] * 10, maxlen=10)
    eeg_buffer = deque([10] * 10, maxlen=10)
    phys_state_buffer = deque(maxlen=5)

    simulate_fatigue = False  # toggle with key

    print("Starting Driver Fatigue Detection... Press 'q' to quit. Press 'f' to toggle fatigue mode.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame.")
            break

        facial_fatigue, face_bbox = detect_facial_fatigue(frame, ear_buffer, mar_buffer)
        avg_ear = np.mean(ear_buffer) if ear_buffer else 0
        avg_mar = np.mean(mar_buffer) if mar_buffer else 0

        # Simulate data
        if simulate_fatigue:
            bpm = np.random.uniform(85, 86)  # Low HRV
            eeg = np.random.normal(16, 0.5)  # High EEG
        else:
            bpm = np.random.uniform(65, 95 if facial_fatigue == 0 else 110)
            eeg = np.random.normal(10 if facial_fatigue == 0 else 15, 1.5)

        bpm_buffer.append(bpm)
        eeg_buffer.append(eeg)

        phys_fatigue, rmssd, avg_eeg = detect_phys_fatigue(bpm_buffer, eeg_buffer, phys_state_buffer)

        # F-score (just a made-up metric for presentation)
        precision = recall = 0.5 + 0.5 * (facial_fatigue + phys_fatigue) / 2
        fscore = 2 * (precision * recall) / (precision + recall)

        state = (facial_fatigue, phys_fatigue)
        frame = declare_fatigue_result(state, frame, face_bbox, avg_ear, avg_mar, rmssd, avg_eeg, fscore)

        cv2.imshow("Driver Fatigue Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            simulate_fatigue = not simulate_fatigue
            print(f"\n[INFO] Fatigue Simulation {'Enabled' if simulate_fatigue else 'Disabled'}")

        time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
