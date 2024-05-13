import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

deepali = face_recognition.load_image_file("2821036.jpg")
deepali_encoding = face_recognition.face_encodings(deepali)[0]
rashika = face_recognition.load_image_file("2821138.jpg")
rashika_encoding = face_recognition.face_encodings(rashika)[0]
aayush = face_recognition.load_image_file("2821137.jpg")
aayush_encoding = face_recognition.face_encodings(aayush)[0]
badal = face_recognition.load_image_file("2821027.jpg")
badal_encoding = face_recognition.face_encodings(badal)[0]

known_face_encodings = [deepali_encoding, rashika_encoding, aayush_encoding, badal_encoding]
known_face_names = ["Deepali", "Rashika", "Aayush", "Badal"]
students = known_face_names.copy()

now = datetime.now()
currentdate = now.strftime("%d-%m-%Y")
f = open(f"{currentdate}.csv", "+w", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    smallframe = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgbsmallf = cv2.cvtColor(smallframe, cv2.COLOR_BGR2XYZ)

    face_locations = face_recognition.face_locations(rgbsmallf)
    face_encodings = face_recognition.face_encodings(rgbsmallf, face_locations)

    for face_encoding in face_encodings:
        match = face_recognition.compare_faces(known_face_encodings, face_encoding)
        facedistance = face_recognition.face_distance(known_face_encodings, face_encoding)
        bestmatchindex = np.argmin(facedistance)

        if match[bestmatchindex]:
            names = known_face_names[bestmatchindex]

            if names in students:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftofCorner = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2
                cv2.putText(frame, names + " Present", bottomLeftofCorner, font, fontScale, fontColor, thickness, lineType)
                students.remove(names)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([names, current_time])

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture, close OpenCV windows, and close CSV file
video_capture.release()
cv2.destroyAllWindows()
f.close()
