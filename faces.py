import cv2              #install opencv-python
import numpy as np      #install numpy
import face_recognition  # install using command  \\\\\     pip install face_recognition  \\\\\   but before it install cmake      \\\\   install dlib using pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp310-cp310-win_amd64.whl   (This is compatible with python 3.10 version)
import os   #This is python's standard library, no need to install seperately
import streamlit as st      # Installing streamlit is very important

st.title("Face Recognition System")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
path = 'images'  #images is the name of the folder where images are kept
images = []
personName = []
myList = os.listdir(path) #myList will now collect the names of all the files present in the path directory


for cu_img in myList:
    current_img = cv2.imread(f'{path}/{cu_img}')    #imread function reads the images from the file
    images.append(current_img)
    personName.append(os.path.splitext(cu_img)[0])   #splittext function splits the filename into 2 parts i.e. name and extension
# print(personName)

def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = faceEncodings(images)
# print(encodeListKnown)
# print("All Encodings Completed!!!")

camera = cv2.VideoCapture(1)

while run:
    ret, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)     #face_locations function returns 4 values which are 4 coordinates of the face box in the selected image
    encodeCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodeCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(matches, faceDis)

        matchIndex = np.argmin(faceDis)
        # print (matchIndex)
        if matches[matchIndex]:
            name = personName[matchIndex]
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 2)
            cv2.rectangle(frame, (x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_TRIPLEX , 1, (0,0,0), 1)
            cv2.putText(frame, f"{round(faceDis[0], 2)}", (x1,y1), cv2.FONT_HERSHEY_TRIPLEX , 1, (255,128,0), 1)

    FRAME_WINDOW.image(frame)

else:
    st.write('Stopped')
