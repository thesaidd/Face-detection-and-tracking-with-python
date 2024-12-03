import cv2
import numpy as np

# Haar cascade yüz algılama modeli
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# PID kontrol parametreleri
P = 0.2  # Daha hızlı hareket için artırın
I = 0.05  # Hataları düzeltme oranını artırabilirsiniz
D = 0.02  # Daha yumuşak hareket için artırın


# PID değişkenleri
prev_error_x = 0
prev_error_y = 0
integral_x = 0
integral_y = 0

# Kamerayı başlat
cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# %30'luk kadraj boyutları
width_offset = int(frame_width * 0.3)
height_offset = int(frame_height * 0.3)

# Başlangıçta kadraj ekranın ortasında
center_x = frame_width // 2
center_y = frame_height // 2

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamıyor!")
        break

    # Siyah çerçeve
    black_frame = np.zeros_like(frame)

    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüz algılama
    faces = face_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.1,  # Daha düşük için 1.05 deneyin
    minNeighbors=5,  # Daha hassas sonuç için artırın
    minSize=(30, 30)  # Minimum yüz boyutunu büyütün
    )


    # Varsayılan hata değerleri
    error_x = 0
    error_y = 0

    if len(faces) > 0:
      #  faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)  # Yüzleri alanlarına göre sırala
        # İlk yüzü al (birden fazla yüz algılanırsa ilkini takip ediyoruz)
        x, y, w, h = faces[0]
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Kadrajın merkezinden yüzün merkezine olan fark
        error_x = face_center_x - center_x
        error_y = face_center_y - center_y

        # PID hesaplamaları
        integral_x += error_x
        integral_y += error_y

        derivative_x = error_x - prev_error_x
        derivative_y = error_y - prev_error_y

        control_x = P * error_x + I * integral_x + D * derivative_x
        control_y = P * error_y + I * integral_y + D * derivative_y

        # Kadraj merkezini güncelle
        center_x += int(control_x)
        center_y += int(control_y)

        

        # Hata değerlerini güncelle
        prev_error_x = error_x
        prev_error_y = error_y

    # Kadraj sınırlarını hesapla (kadrajın ekrandan taşmaması için)
    top = max(0, center_y - height_offset // 2)
    bottom = min(frame_height, center_y + height_offset // 2)
    left = max(0, center_x - width_offset // 2)
    right = min(frame_width, center_x + width_offset // 2)

    # Kadrajın sınırlarını çizin
    cv2.rectangle(black_frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Yeşil bir çerçeve


    # Siyah çerçeve üzerine kadrajı yerleştir
    black_frame[top:bottom, left:right] = frame[top:bottom, left:right]

    # Sonucu göster
    cv2.imshow('Dynamic Frame with PID', black_frame)

    # 'q' tuşuna basıldığında çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
