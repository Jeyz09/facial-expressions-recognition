# collect_data.py
import cv2
import os
import mediapipe as mp

expressions = ['neutral','happy','sad','angry','surprised','fear','disgust','confused','sleepy','excited']
categories = ['human', 'emoji']  # two categories

DATA_PATH = 'data'

# Create folders if not exist
for cat in categories:
    for exp in expressions:
        os.makedirs(os.path.join(DATA_PATH, cat, exp), exist_ok=True)

# Human face data collection using webcam
mp_face = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)
with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
    for exp in expressions:
        print(f"Collecting HUMAN data for: {exp}. Press 'q' to quit, 'n' to next expression.")
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # draw landmarks (optional)
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, face_landmarks, mp_face.FACEMESH_CONTOURS)
                # crop face region
                h, w, _ = frame.shape
                x_min = w; y_min = h; x_max = y_max = 0
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x*w), int(lm.y*h)
                    x_min, y_min = min(x_min,x), min(y_min,y)
                    x_max, y_max = max(x_max,x), max(y_max,y)
                face_img = frame[y_min:y_max, x_min:x_max]
                if face_img.size == 0:
                    continue
                face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                face_gray = cv2.resize(face_gray,(48,48))
                cv2.imwrite(os.path.join(DATA_PATH,'human',exp,f'{count}.png'), face_gray)
                count += 1

            cv2.putText(frame, f'{exp} count: {count}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
            cv2.imshow('Collect Data', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                break
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

cap.release()
cv2.destroyAllWindows()
