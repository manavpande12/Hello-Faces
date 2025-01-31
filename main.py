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
    image_path = guest['image']
    image = cv2.imread(image_path)
    
    # Convert the image to RGB (face_recognition expects RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get face encodings for the guest image
    encodings = face_recognition.face_encodings(rgb_image)
    
    # If a face encoding was found, add it to known faces
    if encodings:
        known_face_encodings.append(encodings[0])
        known_face_names.append(guest['name'])

# Initialize the video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)



# Store the last recognized guest and the timestamp
last_recognized_guest = None
last_recognition_time = 0

while True:
    ret, frame = cap.read()

    if ret:
        # Convert the image from BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each detected face and compare it with known faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found, get the corresponding guest name
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                description = guests[first_match_index]['description']
                
                # Check if the guest was recognized recently (within the last 3 minutes)
                current_time = time.time()
                if name != last_recognized_guest or (current_time - last_recognition_time) > 180:
                    # Print the name and description in the terminal
                    print(f"Name: {name}")
                    print(f"Description: {description}")
                    
                    # Update the last recognized guest and time
                    last_recognized_guest = name
                    last_recognition_time = current_time

            # Draw a rectangle around the face and put the name on top of it
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting image
        cv2.imshow('Face Recognition', frame)

    # Break on pressing 'q'
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
