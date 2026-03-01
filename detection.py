import cv2
from deepface import DeepFace
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import threading
import time

# CSV log file
log_file = "emotion_log.csv"
df = pd.DataFrame(columns=["Time", "Emotion"])
df.to_csv(log_file, index=False)

# Shared variable for thread communication
emotion_counts = {}

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def webcam_loop():
    global emotion_counts
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]

            try:
                result = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']

                # Draw box + label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (36,255,12), 2)

                # Save to CSV
                now = datetime.now().strftime("%H:%M:%S")
                new_data = pd.DataFrame([[now, emotion]], columns=["Time", "Emotion"])
                new_data.to_csv(log_file, mode='a', header=False, index=False)

                # Update counts for live graph
                if emotion not in emotion_counts:
                    emotion_counts[emotion] = 0
                emotion_counts[emotion] += 1

            except:
                pass

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def live_graph():
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,5))

    while True:
        if emotion_counts:
            ax.clear()
            ax.bar(emotion_counts.keys(), emotion_counts.values(), color="skyblue")
            ax.set_title("Real-Time Emotion Frequency")
            ax.set_ylabel("Count")
            plt.draw()
            plt.pause(0.5)
        else:
            time.sleep(0.5)

# Run webcam + graph in parallel
threading.Thread(target=webcam_loop, daemon=True).start()
live_graph()
