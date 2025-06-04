import os
import cv2
import numpy as np
import torch
import time
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp
import sounddevice as sd
import scipy.io.wavfile as wav
from keras.models import load_model
from keras.preprocessing import image as keras_image
import webrtcvad


os.makedirs("evidence/photo", exist_ok=True)
os.makedirs("evidence/audio", exist_ok=True)


face_model = YOLO("models/face-detection.pt")
facenet = InceptionResnetV1(pretrained='vggface2').eval()
spoof_model = load_model("models/face-spoofing.h5")
audio_model = load_model("models/crnn_speech_noise_classifier_improved.h5")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5)
vad = webrtcvad.Vad(2)


SIMILARITY_THRESHOLD = 65.0
BLUR_THRESHOLD = 100.0
SAMPLE_RATE = 16000
VAD_FRAME_DURATION = 30
EVIDENCE_DURATION = 2
UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14
LEFT_EYE_LANDMARKS = [33, 133]
RIGHT_EYE_LANDMARKS = [362, 263]
LEFT_IRIS = [468]
RIGHT_IRIS = [473]
last_spoof_check = 0
last_pose_capture = 0
pose_cooldown = 2

def get_face_embedding(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = face_model(img_rgb)[0]
    if not result.boxes:
        return None
    x1, y1, x2, y2 = map(int, result.boxes.xyxy[0])
    face = img_bgr[y1:y2, x1:x2]
    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).resize((160, 160))
    face_tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float().unsqueeze(0)
    face_tensor = (face_tensor - 127.5) / 128.0
    with torch.no_grad():
        emb = facenet(face_tensor)
    emb = emb.squeeze().numpy()
    return emb / np.linalg.norm(emb)

ref_img = cv2.imread("me.jpeg")
if ref_img is None:
    raise FileNotFoundError("Reference image not found.")
ref_embedding = get_face_embedding(ref_img)


def validate_face(frame):
    embedding = get_face_embedding(frame)
    if embedding is None:
        return "[!] Face not detected", False
    similarity = cosine_similarity([ref_embedding], [embedding])[0][0] * 100
    if similarity < SIMILARITY_THRESHOLD:
        return f"[!] Face mismatch ({similarity:.2f}%)", False
    return f"[✔] Face validated ({similarity:.2f}%)", True

def detect_spoof(frame):
    resized = cv2.resize(frame, (224, 224))
    img_array = keras_image.img_to_array(resized) / 255.0
    prediction = spoof_model.predict(np.expand_dims(img_array, axis=0))[0][0]
    return prediction < 0.5  # True = real face

def get_gaze_direction(eye, iris):
    x1, x2 = eye[0][0], eye[1][0]
    iris_x = iris[0]
    width = abs(x2 - x1)
    if iris_x < x1 + 0.35 * width:
        return "Right"
    elif iris_x > x1 + 0.65 * width:
        return "Left"
    return "Center"

def estimate_head_pose(landmarks, shape):
    h, w, _ = shape
    indices = [33, 263, 1, 61, 291, 199]
    face_3d, face_2d = [], []
    for idx in indices:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])
    face_3d, face_2d = np.array(face_3d, dtype=np.float64), np.array(face_2d, dtype=np.float64)
    focal_length = w
    cam_matrix = np.array([[focal_length, 0, h / 2], [0, focal_length, w / 2], [0, 0, 1]])
    _, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, np.zeros((4, 1)))
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return [a * 360 for a in angles]

def is_mouth_open(landmarks, shape):
    h = shape[0]
    return abs((landmarks[LOWER_LIP_IDX].y - landmarks[UPPER_LIP_IDX].y) * h) > 10

def detect_audio_speech(audio):
    audio_bytes = audio.tobytes()
    frame_length = int(SAMPLE_RATE * VAD_FRAME_DURATION / 1000) * 2
    return any(vad.is_speech(audio_bytes[i:i+frame_length], SAMPLE_RATE)
               for i in range(0, len(audio_bytes), frame_length))

def record_audio():
    audio = sd.rec(int(EVIDENCE_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    return audio

# --- Main Monitoring Loop ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Face validation
    status, valid = validate_face(frame)
    print(status)

    # Spoof check every 10 seconds
    if time.time() - last_spoof_check > 10:
        real_face = detect_spoof(frame)
        print("[✔] Real face detected" if real_face else "[!] Spoof attempt detected")
        if not real_face:
            cv2.imwrite(f"evidence/photo/spoof_{timestamp}.jpg", frame)
        last_spoof_check = time.time()

    # Mediapipe face landmarks
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            h, w, _ = frame.shape

            # Gaze tracking
            le = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE_LANDMARKS]
            re = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE_LANDMARKS]
            li = (int(landmarks[LEFT_IRIS[0]].x * w), int(landmarks[LEFT_IRIS[0]].y * h))
            ri = (int(landmarks[RIGHT_IRIS[0]].x * w), int(landmarks[RIGHT_IRIS[0]].y * h))
            gaze = get_gaze_direction(le, li)
            print(f"[Gaze] {gaze}")

            # Head pose
            x_angle, y_angle, z_angle = estimate_head_pose(landmarks, frame.shape)
            if abs(x_angle) > 30 or abs(y_angle) > 30:
                if time.time() - last_pose_capture > pose_cooldown:
                    cv2.imwrite(f"evidence/photo/headpose_{timestamp}.jpg", frame)
                    print("[!] Suspicious head pose recorded.")
                    last_pose_capture = time.time()

            # Mouth + audio detection
            if is_mouth_open(landmarks, frame.shape):
                print("[!] Mouth open — checking audio...")
                raw_audio = record_audio()
                if detect_audio_speech(raw_audio):
                    photo_path = f"evidence/photo/speech_{timestamp}.jpg"
                    audio_path = f"evidence/audio/speech_{timestamp}.wav"
                    cv2.imwrite(photo_path, frame)
                    wav.write(audio_path, SAMPLE_RATE, raw_audio)
                    print(f"[✔] Speech recorded: {photo_path}, {audio_path}")

    cv2.imshow("Monitoring", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
