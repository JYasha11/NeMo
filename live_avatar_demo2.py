import cv2
import mediapipe as mp
import pyttsx3
import threading
import time

# Initialize mediapipe for face tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Set speech rate
tts_engine.setProperty('volume', 0.9)  # Set volume

# Load avatar and mouth images (adjust paths to your own images)
avatar = cv2.imread('avatar.png')  # Base avatar image
mouth_open = cv2.imread('mouth_open.png')  # Image of mouth open
mouth_closed = cv2.imread('mouth_closed.png')  # Image of mouth closed

# Function to handle TTS in a separate thread
def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# OpenCV capture from webcam
cap = cv2.VideoCapture(0)

# Flag to indicate speaking state
speaking = False

# Function to simulate lip-sync while speaking
def simulate_lip_sync():
    global speaking
    speaking = True
    while speaking:
        time.sleep(0.2)  # Adjust the time delay for lip sync simulation

# Function to overlay the avatar on the video feed with precise positioning
def overlay_avatar(image, avatar, facial_landmarks, size=(200, 200)):
    # Get the coordinates for key facial landmarks
    left_eye = facial_landmarks[33]
    right_eye = facial_landmarks[263]
    nose_tip = facial_landmarks[1]

    # Calculate the center position for the avatar based on facial landmarks
    center_x = int((left_eye[0] + right_eye[0]) / 2)
    center_y = int(nose_tip[1])

    # Calculate the width and height of the avatar based on eye distance
    eye_distance = int(abs(right_eye[0] - left_eye[0]))

    # Resize avatar to match eye distance
    avatar_width = int(eye_distance * 2)  # Scale factor for avatar width
    avatar_height = int(avatar_width * avatar.shape[0] / avatar.shape[1])  # Maintain aspect ratio
    avatar_resized = cv2.resize(avatar, (avatar_width, avatar_height))

    # Position the avatar image
    top_left_x = center_x - avatar_width // 2
    top_left_y = center_y - avatar_height // 2

    # Overlay the avatar onto the frame
    image[top_left_y:top_left_y+avatar_height, top_left_x:top_left_x+avatar_width] = avatar_resized
    return image

# Main loop for processing video frames
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    # Detect face landmarks and adjust avatar position
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get the height and width of the frame
            h, w, _ = image.shape

            # Extract coordinates for key facial landmarks
            facial_landmarks = []
            for idx, landmark in enumerate(face_landmarks.landmark):
                x, y = int(landmark.x * w), int(landmark.y * h)
                facial_landmarks.append((x, y))
                # Draw the landmarks for visualization
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

            # Overlay the avatar based on the detected landmarks
            image = overlay_avatar(image, avatar, facial_landmarks)

    # Display the annotated image
    cv2.imshow('Live Face Tracking', image)

    # Press 'Space' to trigger sample speech
    if cv2.waitKey(5) & 0xFF == 32:  # Space key triggers speech
        threading.Thread(target=speak, args=("Hello, I am your AI avatar!",)).start()
        threading.Thread(target=simulate_lip_sync).start()

    # Exit on pressing 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        speaking = False  # Stop lip-sync when exiting
        break

cap.release()
cv2.destroyAllWindows()
