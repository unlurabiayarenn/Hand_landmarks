import cv2
import mediapipe as mp

# Kamera açma
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

# MediaPipe el takibi modülü
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Sanal nesne (örneğin bir kırmızı top) başlangıç pozisyonu
virtual_object_position = (320, 240)  # Ekranın ortası

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera okuma hatası!")
        break

    # Görüntüyü HSV formatına çevir (renk tespiti için)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Kırmızı renk maskesi oluştur (iki farklı aralıkta)
    lower_red1 = (0, 120, 70)
    upper_red1 = (10, 255, 255)
    lower_red2 = (170, 120, 70)
    upper_red2 = (180, 255, 255)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # El takibi işlemi için BGR görüntüsünü RGB'ye çevir
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # El takibi işlemi
    result = hands.process(frame_rgb)

    # El tespiti varsa, çizim yap
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Elin parmak uçlarının koordinatlarını al
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            # Sanal nesneyi parmak ile aynı pozisyona taşı
            virtual_object_position = (index_x, index_y)

            # Elin hareketlerini çiz
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Sanal nesneyi (kırmızı top) ekranda göster
    cv2.circle(frame, virtual_object_position, 20, (0, 0, 255), -1)

    # Görüntüyü ekranda göster
    cv2.imshow("Sanal Yansima", frame)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
