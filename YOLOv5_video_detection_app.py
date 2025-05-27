# -*- coding: utf-8 -*-
"""
Created on Tue May 27 04:07:41 2025

@author: zzulk
"""

import streamlit as st
import torch
import cv2
import tempfile
import pandas as pd

st.set_page_config(page_title="YOLOv5 Video Detector", layout="centered")
st.title("🦾 YOLOv5 Object Detection on Video")

# Load YOLOv5s model (small version, pretrained on COCO)
@st.cache_resource
def load_yolo_model():
    return torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

model = load_yolo_model()

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])
if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    detection_data = []
    frame_idx = 0

    while cap.isOpened() and frame_idx < 100:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.pandas().xyxy[0]  # bbox, conf, class

        for _, row in detections.iterrows():
            label = f"{row['name']} {row['confidence']:.2f}"
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            detection_data.append({
                "frame": frame_idx,
                "class": row["name"],
                "confidence": row["confidence"],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })

        stframe.image(frame, channels="BGR", use_column_width=True)
        frame_idx += 1

    cap.release()
    df = pd.DataFrame(detection_data)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Detections CSV", data=csv, file_name="detections.csv")
    st.success("✅ Detection complete!")
