import cv2
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
face_mesh = mp_face.FaceMesh(static_image_mode=False,
                             max_num_faces=1,
                             refine_landmarks=True,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

# Setup webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Face tracking started. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw facial landmarks (for visualization)
            mp_draw.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_TESSELATION,
                                   landmark_drawing_spec=None,
                                   connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))

            # Extract nose tip coordinates (landmark 1)
            nose = face_landmarks.landmark[1]
            nose_x, nose_y = int(nose.x * w), int(nose.y * h)
            cv2.circle(frame, (nose_x, nose_y), 5, (0, 0, 255), -1)

            # Find bounding box
            x_min = min([int(lm.x * w) for lm in face_landmarks.landmark])
            y_min = min([int(lm.y * h) for lm in face_landmarks.landmark])
            x_max = max([int(lm.x * w) for lm in face_landmarks.landmark])
            y_max = max([int(lm.y * h) for lm in face_landmarks.landmark])

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Print nose position (normalized 0–1)
            print(f"Nose position (normalized): x={nose.x:.3f}, y={nose.y:.3f}")

    cv2.imshow("Face Tracking Demo", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
