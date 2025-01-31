import cv2
import face_recognition
import json
import time
import pyttsx3
import threading
import queue

speech_queue = queue.Queue()
spoken_names = {}
last_spoken_guest = None  # Track the last guest spoken
last_spoken_time = 0  # Track the last spoken time globally

def speak_worker():
    engine = pyttsx3.init('sapi5')
    Id = r'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0'
    engine.setProperty('voice', Id)
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

def speak(text):
    if text:
        speech_queue.put(text)

speech_thread = threading.Thread(target=speak_worker, daemon=True)
speech_thread.start()

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

# Store the last recognized guests and timestamps
while True:
    ret, frame = cap.read()
    
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        current_time = time.time()

        if len(face_locations) == 1:  # Check if there is only one face detected
            top, right, bottom, left = face_locations[0]  # Get the first (and only) face location
            face_encoding = face_encodings[0]  # Get the first (and only) face encoding

            # Check for the best match
            name = "Unknown"
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            if True in matches:
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                guest_info = next((g for g in guests if g['name'] == name), None)
                description = guest_info['description'] if guest_info else "No description available"

                # Reset cooldown if a new guest is detected
                if name != last_spoken_guest or (current_time - last_spoken_time) > 180:
                    speak(f"Name: {name}")
                    speak(f"Description: {description}")
                    last_spoken_guest = name
                    last_spoken_time = current_time

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            # If more than one face is detected, do nothing
            pass

        cv2.imshow('Hello Faces', frame)

    # Break on pressing 'q'
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

speech_queue.put(None)  # Stop the speech thread
speech_thread.join()
cap.release()
cv2.destroyAllWindows()
