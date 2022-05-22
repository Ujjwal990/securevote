import json
import cv2              #install opencv-python
import numpy as np      #install numpy
import face_recognition  # install using command  \\\\\     pip install face_recognition  \\\\\   but before it install cmake      \\\\   install dlib using pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp310-cp310-win_amd64.whl   (This is compatible with python 3.10 version)
import os   #This is python's standard library, no need to install seperately
import streamlit as st      # Installing streamlit is very important
                            #install streamlit-lottie using     pip install streamlit-lottie
                            #pip install requests
import requests
from streamlit_lottie import st_lottie
# from PIL import Image
st. set_page_config(layout="wide")
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def local_css(file_name):   #using local css
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style/style.css")


st.title("*Welcome to Face Recognition System* ")

run = st.checkbox('  START / STOP recognition')
st.markdown('It is $$ Really Cool$$.')

lottie_coding = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_2szpas4y.json")





############################################

# st.write(":smile:"*4)

FRAME_WINDOW = st.image([])
path = 'photos'  #images is the name of the folder where images are kept
images = []
personName = []
myList = os.listdir(path) #myList will now collect the names of all the files present in the path directory


for cu_img in myList:
    current_img = cv2.imread(f'{path}/{cu_img}')    #imread function reads the images from the file
    images.append(current_img)
    personName.append(os.path.splitext(cu_img)[0])   #splittext function splits the filename into 2 parts i.e. name and extension

def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = faceEncodings(images)

camera = cv2.VideoCapture(1)

while run:
    ret, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)     #face_locations function returns 4 values which are 4 coordinates of the face box in the selected image
    encodeCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodeCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)       #it return a boolean value i.e. True or False
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)       #It denoted the dissimilarity between two faces

        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = personName[matchIndex]
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 2)   
            cv2.rectangle(frame, (x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_TRIPLEX , 1, (0,0,0), 1)
            cv2.putText(frame, f"{round(faceDis[0], 2)}", (x1,y1), cv2.FONT_HERSHEY_TRIPLEX , 1, (255,128,0), 1)
            cv2.putText(frame, f"match found", (x1+65,y1-10), cv2.FONT_HERSHEY_TRIPLEX , 0.5, (255,255,255), 1)


    FRAME_WINDOW.image(frame)

with st.container():
    st.write("---")
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("How it works")
        st.write("##")
        st.write(
            """
            - It leverage the power of Python and streamlit to recognize the faces of people.
            - Dlib library marks 128 key points and stores those values for comparing them. 
            - face_recognition library then compares those 128 Unique identification points and returns whether the face is matched or not.
            - face_recognition.compare_faces() function compares the two faces and produces the result i.e. true or false.
            - face_recognition.face_distance() function gives the amount of similarity in two faces which is between 0 and 1.
            """
        )
        
    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")

#####################################

with st.container():
    st.write("---")
    st.header("Contact Developer")
    st.write("##")

    contact_form = """
    

    <form action="https://formsubmit.co/capitalgabru@gmail.com
" autocomplete="off" method = "POST">
            <h3 class="title">Contact us</h3>
            <div class="input-container">
              <input type="text" name="name" class="input" />
              <label for="">Username</label>
              <span>Username</span>
            </div>
            <div class="input-container">
              <input type="email" name="email" class="input" />
              <label for="">Email</label>
              <span>Email</span>
            </div>
            <div class="input-container">
              <input type="tel" name="phone" class="input" />
              <label for="">Phone</label>
              <span>Phone</span>
            </div>
            <div class="input-container textarea">
              <textarea name="message" class="input"></textarea>
              <label for="">Message</label>
              <span>Message</span>
            </div>
            <input type="submit" value="Send" class="btn" />
    </form>

    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()
