import cv2       # OpenCV 라이브러리 임포트
import numpy as np  # 행렬 연산을 위한 NumPy

def edge_detection_pipeline():
    # 1. 웹캠 연결 (0번은 내장 웹캠)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    cv2.namedWindow('Controls')
    cv2.createTrackbar('Low Threshold', 'Controls', 30, 50, lambda x: None)
    cv2.createTrackbar('High Threshold', 'Controls', 90, 150, lambda x: None)

    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        low = cv2.getTrackbarPos('Low Threshold', 'Controls')
        high = cv2.getTrackbarPos('High Threshold', 'Controls')

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])

        lower_red1 = np.array([0,   120,  70])
        upper_red1 = np.array([10,  255, 255])
        lower_red2 = np.array([170, 120,  70])
        upper_red2 = np.array([180, 255, 255])



        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # masked = cv2.bitwise_and(gray, gray, mask=mask)
        edges = cv2.Canny(mask, low, high)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 700:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # edges = cv2.Canny(blurred, low, high)

        cv2.imshow('Frame Video', frame)
        # cv2.imshow('Edges Video', edges)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

# 실행
edge_detection_pipeline()