import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt') # Nano 모델이 속도가 가장 빠름
cap = cv2.VideoCapture(0) # 0은 기본 웹캠

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    # stream=True 옵션은 메모리 효율적인 제너레이터를 반환
    results = model.predict(frame, stream=True)
    
    for r in results:
        annotated_frame = r.plot() # OpenCV 호환 배열 반환
        cv2.imshow("YOLOv8 Real-time Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' 키로 종료
        break
cap.release()
cv2.destroyAllWindows()