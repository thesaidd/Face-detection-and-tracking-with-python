import cv2
import numpy as np  # NumPy kütüphanesini ekleyin

# OpenCV'nin önceden eğitilmiş Haar cascade yüz algılama modelini yükleyin.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kamerayı başlat
cap = cv2.VideoCapture(0)  # 0, varsayılan kamera anlamına gelir.

while True:
    # Kameradan bir kare al
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamıyor!")
        break

    # Görüntüyü gri tonlamaya çevir (algılama için gereklidir)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Tespit edilen her yüz için çerçeve çiz
    for (x, y, w, h) in faces:
        # Yüz çevresine kare çiz
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Yüzün merkezini hesapla
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Kameranın merkezini hesapla
        frame_height, frame_width, _ = frame.shape
        camera_center_x = frame_width // 2
        camera_center_y = frame_height // 2

        # Yüzün orta noktasından kameranın orta noktasına vektör çiz
        cv2.line(frame, (camera_center_x, camera_center_y), (face_center_x, face_center_y), (0, 0, 255), 2)  # Kırmızı renkli vektör

        # Yüz bölgesini almak için orijinal kareyi siyah bir arka planla yenileyin
        black_frame = np.zeros_like(frame)  # NumPy kullanarak siyah arka plan oluşturuyoruz
        black_frame[y:y+h, x:x+w] = frame[y:y+h, x:x+w]  # Yüz bölgesini koru

        frame = black_frame  # Yenilenmiş siyah arka planlı görüntüyü kullan

    # Sonucu göster
    cv2.imshow('Face Tracking with Black Background', frame)

    # 'q' tuşuna basılarak çıkış yapılabilir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()