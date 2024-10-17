import cv2
import numpy as np
import face_recognition
import mediapipe

def generate_simple_avatar(size):
    avatar = np.zeros((size, size, 3), dtype=np.uint8)
    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    cv2.circle(avatar, (size // 2, size // 2), size // 2, color, -1)
    return avatar

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert frame to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find face locations
    face_locations = face_recognition.face_locations(rgb_frame)

    for (top, right, bottom, left) in face_locations:
        face_height = bottom - top
        face_width = right - left
        
        # Generate a simple avatar for each face
        avatar = generate_simple_avatar(max(face_height, face_width))
        
        # Resize avatar to match face size
        avatar = cv2.resize(avatar, (face_width, face_height))
        
        # Create a mask for smooth blending
        mask = np.ones(avatar.shape[:2], dtype=np.float32)
        mask = cv2.GaussianBlur(mask, (int(face_width/4) * 2 + 1, int(face_height/4) * 2 + 1), 0)
        mask = np.dstack([mask] * 3)
        
        # Ensure we're not going out of bounds
        y1, y2 = max(0, top), min(frame.shape[0], bottom)
        x1, x2 = max(0, left), min(frame.shape[1], right)
        
        # Adjust avatar and mask size if needed
        avatar = avatar[:y2-y1, :x2-x1]
        mask = mask[:y2-y1, :x2-x1]
        
        # Blend avatar with original frame
        frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - mask) + avatar * mask

    # Display the result
    cv2.imshow('AI Avatar Overlay', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()