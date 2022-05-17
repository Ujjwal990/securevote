import cv2 #This opencv module is used to access the camera
import streamlit as st

st.title("My Web Application")
run = st.checkbox('Run')        #Here run is a variable or object
Frame_Window = st.image([])
cam = cv2.VideoCapture(0)  #Here '0' means that laptop's primary webcam will be used and '1' means using other connected camera

while run:
    ret, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #COLOR_BGR2RGB is converting BGR format to RGB format video/image
    Frame_Window.image(frame)
else:
    st.write('cannot take input from camera')

