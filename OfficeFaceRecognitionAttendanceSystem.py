# Made By Nishant Jangid
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
print("Made By Nishant Jangid")

# Loading the Images From ImagesAttendance Directory
path = 'images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Finding the Encoding of the Faces
def findEncodings(images):
    encodeList =[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Marking the Attendance in the Attendance.csv File


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        # Read existing data
        myDataList = f.readlines()
        nameList = [entry.split(',')[0] for entry in myDataList]

        time_now = datetime.now()
        tString = time_now.strftime('%H:%M:%S')
        dString = time_now.strftime('%d/%m/%Y')

        entry_time_start = datetime.strptime("17:00:00", "%H:%M:%S").time()
        entry_time_end = datetime.strptime("17:59:00", "%H:%M:%S").time()
        exit_time_start = datetime.strptime("18:00:00", "%H:%M:%S").time()
        exit_time_end = datetime.strptime("19:30:00", "%H:%M:%S").time()

        f.seek(0, 2)  # Move the cursor to the end of the file

        if name not in nameList:  # Entry
            if entry_time_start <= time_now.time() <= entry_time_end:
                # Record entry time in the 'Entry' column
                f.write(f'\n{name},Entry,{tString},{dString},')
            else:
                # Record exit time in the 'Exit' column
                f.write(f'\n{name},Exit,{tString},{dString},')
        else:  # Exit (assuming only one entry per day)
            for i, line in enumerate(myDataList):
                entry = line.split(',')
                if entry[0] == name and entry[-1].strip() == "":
                    if exit_time_start <= time_now.time() <= exit_time_end:
                        # Check if "Exit" is already in the line
                        if "Exit" not in line:
                            # Append exit time in the same row without overwriting entry details
                            myDataList[i] = f'{line.strip()},Exit,{tString},{dString}\n'
                            f.seek(0)
                            f.writelines(myDataList)
                    break



encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Capturing the images of peope using webcam
cap = cv2.VideoCapture(0)

name2= " Unkown "

# Comparing the Captured Images with the Images in the database or the ImagesAttendance Directory
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 250, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 250, 0), cv2.FILLED)
            cv2.putText(img, name2 ,(x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            print("Unkown")
    cv2.imshow('webcam', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print(" Made By Nishant Jangid ")

