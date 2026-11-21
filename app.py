
import streamlit as st
import cv2
import tempfile
import os
from collections import Counter
from ultralytics import YOLO

st.set_page_config(
    page_title="RoadVision AI",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 RoadVision AI")
st.subheader("Intelligent Vehicle Detection & Tracking System")

st.write(
    "Detect and track cars, buses, trucks, people and other objects "
    "using AI-powered computer vision."
)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.sidebar.header("Settings")
confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=1.00,
    value=0.35,
    step=0.05
)

mode = st.radio(
    "Choose Detection Mode",
    ["Image Detection", "Video Detection & Tracking"],
    horizontal=True
)

if mode == "Image Detection":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        file_bytes = uploaded_file.read()
        image = cv2.imdecode(
            __import__("numpy").frombuffer(file_bytes, dtype="uint8"),
            cv2.IMREAD_COLOR
        )

        if st.button("Detect Objects"):
            with st.spinner("AI is detecting objects..."):
                results = model.predict(
                    image,
                    conf=confidence,
                    verbose=False
                )

                annotated = results[0].plot()
                annotated = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    annotated,
                    caption="Detection Results",
                    use_container_width=True
                )

                names = results[0].names
                counts = Counter(
                    names[int(box.cls[0])]
                    for box in results[0].boxes
                )

                st.subheader("Detection Summary")
                col1, col2 = st.columns(2)
                col1.metric("Total Objects", sum(counts.values()))
                col2.metric("Object Categories", len(counts))

                for name, count in counts.items():
                    st.write(f"**{name.title()}:** {count}")

elif mode == "Video Detection & Tracking":
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video:
        if st.button("Start Detection & Tracking"):
            temp_input = None
            temp_output = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as input_file:
                    input_file.write(uploaded_video.read())
                    temp_input = input_file.name

                with st.spinner(
                    "Detecting and tracking objects in video..."
                ):
                    cap = cv2.VideoCapture(temp_input)

                    if not cap.isOpened():
                        st.error("Could not open this video.")
                    else:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)

                        if not fps or fps <= 0:
                            fps = 25

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        ) as output_file:
                            temp_output = output_file.name

                        writer = cv2.VideoWriter(
                            temp_output,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (width, height)
                        )

                        frame_placeholder = st.empty()
                        progress = st.progress(0)
                        status = st.empty()

                        total_frames = int(
                            cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        )
                        frame_number = 0
                        track_ids = set()
                        object_counts = Counter()

                        while cap.isOpened():
                            success, frame = cap.read()

                            if not success:
                                break

                            results = model.track(
                                frame,
                                persist=True,
                                tracker="bytetrack.yaml",
                                conf=confidence,
                                verbose=False
                            )

                            annotated = results[0].plot()
                            writer.write(annotated)

                            boxes = results[0].boxes

                            if boxes is not None:
                                for box in boxes:
                                    class_id = int(box.cls[0])
                                    object_counts[
                                        results[0].names[class_id]
                                    ] += 1

                                    if box.id is not None:
                                        track_ids.add(
                                            int(box.id[0])
                                        )

                            frame_number += 1

                            if frame_number % 3 == 0:
                                rgb = cv2.cvtColor(
                                    annotated,
                                    cv2.COLOR_BGR2RGB
                                )
                                frame_placeholder.image(
                                    rgb,
                                    channels="RGB",
                                    use_container_width=True
                                )

                            if total_frames > 0:
                                progress.progress(
                                    min(frame_number / total_frames, 1.0)
                                )

                            status.write(
                                f"Processing frame {frame_number} "
                                f"of {total_frames}"
                            )

                        cap.release()
                        writer.release()

                        st.success("Video processing completed!")

                        st.subheader("Tracking Summary")
                        col1, col2 = st.columns(2)
                        col1.metric(
                            "Unique Tracked Objects",
                            len(track_ids)
                        )
                        col2.metric(
                            "Processed Frames",
                            frame_number
                        )

                        st.write("**Detected object classes:**")
                        for name, count in object_counts.items():
                            st.write(f"- {name.title()}: {count} detections")

                        if temp_output and os.path.exists(temp_output):
                            with open(temp_output, "rb") as video_file:
                                st.download_button(
                                    "Download Processed Video",
                                    data=video_file.read(),
                                    file_name="roadvision_output.mp4",
                                    mime="video/mp4"
                                )

            except Exception as error:
                st.error(f"Something went wrong: {error}")

            finally:
                if temp_input and os.path.exists(temp_input):
                    os.remove(temp_input)
                # Keep output available until the download button is used.
                # Streamlit reruns the script after download interaction.
