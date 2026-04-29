import cv2
import numpy as np
from ultralytics import YOLO

seg_model  = YOLO('yolov8n-seg.pt')
pose_model = YOLO('yolov8n-pose.pt')

cap = cv2.VideoCapture(0)

THUMB_W, THUMB_H = 160, 120
COLS = 3

# COCO 스켈레톤 연결 쌍
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),          # 얼굴
    (5,6),(5,7),(7,9),(6,8),(8,10),   # 팔
    (5,11),(6,12),(11,12),            # 몸통
    (11,13),(13,15),(12,14),(14,16),  # 다리
]

def draw_pose(frame, pose_results):
    for r in pose_results:
        if r.keypoints is None:
            continue
        for kp in r.keypoints:
            pts = kp.xy[0].cpu().numpy().astype(int)   # (17, 2)
            conf = kp.conf[0].cpu().numpy()            # (17,)

            for a, b in SKELETON:
                if conf[a] > 0.4 and conf[b] > 0.4:
                    cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 255, 255), 2)

            for i, (x, y) in enumerate(pts):
                if conf[i] > 0.4:
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    seg_results  = seg_model.predict(frame,  stream=True, verbose=False)
    pose_results = pose_model.predict(frame, stream=True, verbose=False)

    crops = []
    annotated = frame.copy()

    for r in seg_results:
        annotated = r.plot()

        if r.boxes is None:
            continue

        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2].copy()

            if r.masks is not None and i < len(r.masks):
                mask = r.masks.data[i].cpu().numpy()
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_crop = (mask[y1:y2, x1:x2] * 255).astype(np.uint8)
                crop = cv2.bitwise_and(crop, crop, mask=mask_crop)

            cls_id = int(box.cls[0])
            label  = f"{seg_model.names[cls_id]} {float(box.conf[0]):.2f}"
            crop_resized = cv2.resize(crop, (THUMB_W, THUMB_H))
            cv2.putText(crop_resized, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            crops.append(crop_resized)

    # 스켈레톤을 annotated 위에 덧그림
    draw_pose(annotated, pose_results)

    if crops:
        rows = (len(crops) + COLS - 1) // COLS
        grid = np.zeros((rows * THUMB_H, COLS * THUMB_W, 3), dtype=np.uint8)
        for idx, crop in enumerate(crops):
            r_i, c_i = divmod(idx, COLS)
            grid[r_i*THUMB_H:(r_i+1)*THUMB_H, c_i*THUMB_W:(c_i+1)*THUMB_W] = crop
        cv2.imshow("Detected Objects", grid)

    cv2.imshow("YOLOv8 Segmentation + Pose", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
