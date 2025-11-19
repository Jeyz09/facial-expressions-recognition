# recognize.py
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from train import FER_Emoji_CNN, expressions  # make sure both scripts in same folder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FER_Emoji_CNN(num_expressions=len(expressions)).to(device)
model.load_state_dict(torch.load('fer_emoji_model.pth', map_location=device))
model.eval()

cap = cv2.VideoCapture(0)
mp_face = mp.solutions.face_mesh

with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Bounding box
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
                face_tensor = torch.tensor(face_gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0)/255.0
                face_tensor = (face_tensor-0.5)/0.5
                face_tensor = face_tensor.to(device)
                with torch.no_grad():
                    cat_out, expr_out = model(face_tensor)
                    category = torch.argmax(cat_out,1).item()  # 0=Human, 1=Emoji
                    expression = torch.argmax(expr_out,1).item()
                label_text = f"{'Human' if category==0 else 'Emoji'}: {expressions[expression]}"
                cv2.putText(frame, label_text, (x_min,y_min-10), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow("FER + Emoji Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
