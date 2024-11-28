import cv2
import pytesseract
from ultralytics import YOLO

# بارگذاری مدل YOLOv8
model = YOLO("best.pt")  # جایگزین با مدل آموزش‌داده‌شده

# باز کردن دوربین
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # تشخیص پلاک‌ها با YOLOv8
    results = model(frame)
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = map(int, result[:6])
        
        # برش پلاک
        plate_img = frame[y1:y2, x1:x2]
        
        # استخراج متن از تصویر
        plate_number = pytesseract.image_to_string(plate_img, lang='eng+ara')  # پلاک ایرانی: اعداد و حروف عربی
        print("شماره پلاک:", plate_number.strip())

        # نمایش نتیجه
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, plate_number.strip(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
