
import cv2
import mediapipe as mp
import numpy as np

# Mediapipe Hand Tracking modülü
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Renk kutuları (HSV eşikleri ile birlikte)
colors = [
    {"name": "Blue", "rgb": (255, 0, 0), "hsv": ((100, 150, 0), (140, 255, 255))},
    {"name": "Green", "rgb": (0, 255, 0), "hsv": ((40, 70, 0), (90, 255, 255))},
    {"name": "Red", "rgb": (0, 0, 255), "hsv": ((0, 120, 70), (10, 255, 255))},
    {"name": "Yellow", "rgb": (0, 255, 255), "hsv": ((20, 100, 100), (30, 255, 255))}
]

selected_color = None
selected_color_name = None
selected_box_coords = None  # Seçilen kutunun koordinatları

# Kamera başlatma
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Aynalama
        h, w, _ = frame.shape

        # Renk kutularını çiz
        num_colors = len(colors)
        box_width = w // num_colors  # Eşit genişlikte kutular
        box_height = h // 8  # Sabit yükseklik

        for i, color in enumerate(colors):
            x1 = i * box_width
            y1 = 0
            x2 = (i + 1) * box_width
            y2 = box_height
            cv2.rectangle(frame, (x1, y1), (x2, y2), color["rgb"], -1)
            cv2.putText(frame, color["name"], (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Mediapipe el algılama
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # İşaret parmağının ucu (8. landmark)
                x_index = int(hand_landmarks.landmark[8].x * w)
                y_index = int(hand_landmarks.landmark[8].y * h)

                # İşaret parmağının ucuna daire ekle
                cv2.circle(frame, (x_index, y_index), 10, (0, 0, 255), -1)  # Kırmızı daire
                cv2.circle(frame, (x_index, y_index), 15, (255, 255, 255), 2)  # Beyaz kontur

                # Parmağın renk seçme bölgesinde olup olmadığını kontrol et
                for i, color in enumerate(colors):
                    x1 = i * box_width
                    y1 = 0
                    x2 = (i + 1) * box_width
                    y2 = box_height
                    if x1 < x_index < x2 and y1 < y_index < y2:
                        selected_color = color
                        selected_color_name = color["name"]
                        selected_box_coords = (x1, y1, x2, y2)  # Seçilen kutunun koordinatları
                        break

        # Eğer bir renk seçildiyse, ekrandaki o renk nesnelerini algıla
        if selected_color:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_hsv, upper_hsv = selected_color["hsv"]

            # Maske oluştur
            mask = cv2.inRange(hsv_frame, np.array(lower_hsv), np.array(upper_hsv))
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # Konturları bul
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Minimum alan filtresi
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), selected_color["rgb"], 2)

                    # İşaret parmağının çerçeveyle kesişip kesişmediğini kontrol et
                    if x < x_index < x + w and y < y_index < y + h:
                        # Eğer çerçeveyle kesişiyorsa, doğru renk mesajını ekle
                        cv2.putText(frame, "DOĞRU RENK", (50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

        cv2.imshow("Color Selection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
