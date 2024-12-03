import cv2
import numpy as np

# PID kontrol parametreleri
P = 0.2  # Daha hızlı hareket için artırın
I = 0.05  # Hataları düzeltme oranını artırabilirsiniz
D = 0.02  # Daha yumuşak hareket için artırın

# PID değişkenleri
prev_error_x = 0
prev_error_y = 0
integral_x = 0
integral_y = 0

# Yüz algılama için Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    kadraj_width = int(frame.shape[1] * 0.3)  # Varsayılan %30 genişlik
    kadraj_height = int(frame.shape[0] * 0.3)  # Varsayılan %30 yükseklik

    # Kamera merkezi
    cam_center_x, cam_center_y = frame.shape[1] // 2, frame.shape[0] // 2

    # En yakın yüzü seç
    if len(faces) > 0:
        # Mesafeye göre sırala (yüz merkezi, kamera merkezi mesafesi)
        faces = sorted(faces, key=lambda f: ((f[0] + f[2] // 2 - cam_center_x) ** 2 + (f[1] + f[3] // 2 - cam_center_y) ** 2))
        x, y, w, h = faces[0]  # En yakın yüz
        face_center_x, face_center_y = x + w // 2, y + h // 2

        # Kamera merkezi ile yüz merkezi arasında vektör çiz
        frame = cv2.arrowedLine(
            frame,
            (cam_center_x, cam_center_y),
            (face_center_x, face_center_y),
            (0, 255, 0), 2
        )

        # Kadraj sınırlarını ayarla (yüzü merkezde tutmaya çalış)
        left = max(0, face_center_x - kadraj_width // 2)
        right = min(frame.shape[1], face_center_x + kadraj_width // 2)
        top = max(0, face_center_y - kadraj_height // 2)
        bottom = min(frame.shape[0], face_center_y + kadraj_height // 2)
    else:
        # Yüz yoksa varsayılan kadraj
        left = max(0, cam_center_x - kadraj_width // 2)
        right = min(frame.shape[1], cam_center_x + kadraj_width // 2)
        top = max(0, cam_center_y - kadraj_height // 2)
        bottom = min(frame.shape[0], cam_center_y + kadraj_height // 2)

    # Kadraj dışında kalan yerleri siyah yap
    black_frame = np.zeros_like(frame)
    black_frame[top:bottom, left:right] = frame[top:bottom, left:right]

    # Görüntüyü göster
    cv2.imshow("Face Tracking", black_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()