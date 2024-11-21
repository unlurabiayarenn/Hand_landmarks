import cv2
import numpy as np
import random
import mediapipe as mp

# Mediapipe modülleri
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 4 ana renk (HSV formatında)
colors = [
    {"name": "Red", "hsv_lower": np.array([0, 120, 70]), "hsv_upper": np.array([10, 255, 255])},
    {"name": "Green", "hsv_lower": np.array([40, 70, 70]), "hsv_upper": np.array([90, 255, 255])},
    {"name": "Blue", "hsv_lower": np.array([100, 150, 0]), "hsv_upper": np.array([140, 255, 255])},
    {"name": "Yellow", "hsv_lower": np.array([20, 100, 100]), "hsv_upper": np.array([30, 255, 255])}
]

# Kamera başlat
cap = cv2.VideoCapture(0)

# Rastgele bir renk seç
random_color = random.choice(colors)

# Mediapipe el tespiti başlat
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Aynalama

        # Renkleri HSV formatına dönüştür
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Renklerin tespit edilip edilmediğini kontrol et
        detected_colors = []

        for color in colors:
            mask = cv2.inRange(hsv_frame, color["hsv_lower"], color["hsv_upper"])

            # Eğer renk tespit edilirse, renk ismini detected_colors listesine ekle
            if np.any(mask):
                detected_colors.append(color["name"])

        # Mediapipe el algılama
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # İşaret parmağı ucu (8. landmark)
                x_index = int(hand_landmarks.landmark[8].x * frame.shape[1])
                y_index = int(hand_landmarks.landmark[8].y * frame.shape[0])

                # İşaret parmağının ucunu belirginleştir
                cv2.circle(frame, (x_index, y_index), 10, (0, 0, 255), -1)

        # Sağ üst köşeye rastgele seçilen rengi yaz
        cv2.putText(frame, f"Color: {random_color['name']}", (frame.shape[1] - 250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)

        # Ekranı göster
        cv2.imshow("Color and Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
