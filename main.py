from fastapi import FastAPI, File, UploadFile
import cv2
import tempfile
import mediapipe as mp

app = FastAPI()
mp_face = mp.solutions.face_detection

@app.post("/check-face/")
async def check_face(video: UploadFile = File(...)):
    # Save uploaded video to temp file
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp.write(await video.read())
    temp.close()

    cap = cv2.VideoCapture(temp.name)
    detector = mp_face.FaceDetection(min_detection_confidence=0.5)

    frame_count, face_count = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        if results.detections:
            face_count += 1

    cap.release()

    return {
        "valid_face": face_count >= (0.8 * frame_count),  # Face must appear in 80% frames
        "frames_checked": frame_count,
        "faces_detected_in_frames": face_count
    }
