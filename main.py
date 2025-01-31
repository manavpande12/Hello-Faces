import cv2
import face_recognition
import json
import time

# Load guest information from the JSON file
with open('guests.json', 'r') as f:
    guests = json.load(f)

# Initialize known faces and labels
known_face_encodings = []
known_face_names = []

# Load images and create embeddings for each guest
for guest in guests:
    for image_path in guest['images']:  # Loop through all images per guest
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_image)

        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(guest['name'])  # Store the same name for all images

# Initialize the video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Store the last recognized guest and timestamp
last_recognized_guest = None
last_recognition_time = 0

while True:
    ret, frame = cap.read()
    
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Find best match
            if True in matches:
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                guest_info = next((g for g in guests if g['name'] == name), None)
                description = guest_info['description'] if guest_info else "No description available"

                # Check if recognized within the last 3 minutes
                current_time = time.time()
                if name != last_recognized_guest or (current_time - last_recognition_time) > 180:
                    print(f"Name: {name}")
                    print(f"Description: {description}")
                    last_recognized_guest = name
                    last_recognition_time = current_time

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Hello Faces', frame)

    # Break on pressing 'q'
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
