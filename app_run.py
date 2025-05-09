import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
import os
import torch

# Load YOLOv8 model and force GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("toy_model_results/yolov8s_toy11/weights/best.pt").to(device)

# Display device info in sidebar
st.sidebar.title("YOLOv8 Settings")
st.sidebar.write(f"Running on: **{device.upper()}**")

st.title("YOLOv8 Toy Detection App")
mode = st.sidebar.radio("Choose input source", ["Upload Video", "Use Webcam"])

def infer_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame)
        annotated_frame = results[0].plot()

        if frame_count % 3 == 0:
            resized_frame = cv2.resize(annotated_frame, (640, 360))
            stframe.image(resized_frame, channels="BGR", use_column_width=True)

    cap.release()

def infer_on_webcam():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    frame_count = 0
    stop_clicked = False

    stop_button = st.button("Stop Webcam")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame)
        annotated_frame = results[0].plot()

        if frame_count % 3 == 0:
            resized_frame = cv2.resize(annotated_frame, (640, 360))
            stframe.image(resized_frame, channels="BGR", use_column_width=True)

        if stop_button:
            break

    cap.release()

# --- Video Upload Mode ---
if mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.flush()
        video_path = tfile.name
        st.success("Video uploaded successfully!")

        if st.button("Run Inference"):
            infer_on_video(video_path)
            cap = cv2.VideoCapture(video_path)
            cap.release()
            os.unlink(video_path)

# --- Webcam Mode ---
elif mode == "Use Webcam":
    if st.button("Start Webcam"):
        infer_on_webcam()
