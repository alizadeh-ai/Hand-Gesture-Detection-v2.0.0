import cv2
import mediapipe as mp
import time
import autopy

# MediaPipe Tasks Setup =>
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# Webcam =>
cap = cv2.VideoCapture(0)

prev_time = 0
current_time = 0
frame_id = 0

while True:
    success, img = cap.read()
    if not success:
        break

    frame_id += 1

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

    results = landmarker.detect_for_video(mp_image, frame_id)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:

            point8 = 0
            point4 = 0
            point12 = 0

            h, w, c = img.shape

            for id, lm in enumerate(hand_landmarks):

                cx, cy = int(lm.x * w), int(lm.y * h)

                print(id, ": ", cx, cy)

                if id == 8:
                    cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)
                    cx8 = cx
                    cy8 = cy
                    point8 = (cx + cy)
                    autopy.mouse.move(cx8, cy8)

                if id == 4:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 0), cv2.FILLED)
                    point4 = (cx + cy)

                if id == 12:
                    cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)
                    point12 = (cx + cy)

                if point8 and point4 and point12:
                    if abs(point12 - point4) < 30:
                        cv2.putText(img, "right click", (50, 50),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                        autopy.mouse.click(autopy.mouse.Button.RIGHT)

                    if abs(point8 - point4) < 30:
                        cv2.putText(img, "left click", (50, 50),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                        autopy.mouse.click(autopy.mouse.Button.LEFT)

            
            for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                start_idx = connection.start
                end_idx = connection.end

                x0 = int(hand_landmarks[start_idx].x * w)
                y0 = int(hand_landmarks[start_idx].y * h)
                x1 = int(hand_landmarks[end_idx].x * w)
                y1 = int(hand_landmarks[end_idx].y * h)

                cv2.line(img, (x0, y0), (x1, y1), (255, 255, 255), 2)

    # FPS =>
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
