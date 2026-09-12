import os

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import tempfile
import subprocess

from collections import deque, Counter


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Driver Behavior Analysis",
    page_icon="🚗",
    layout="wide"
)


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "driver_behavior_efficientnet.keras"

CLASS_NAMES = [
    "other_activities",
    "safe_driving",
    "talking_phone",
    "texting_phone",
    "turning"
]

IMG_SIZE = 224

SMOOTHING_WINDOW = 15

PREDICT_EVERY_N_FRAMES = 2


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .video-title {
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🚗 Driver Behavior Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Driver Behavior Detection using EfficientNet'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


try:

    model = load_model()

except Exception as e:

    st.error("❌ Could not load the model.")
    st.code(str(e))
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Model Information")

st.sidebar.write(
    "Architecture: EfficientNetB0"
)

st.sidebar.write(
    "Task: Driver Behavior Classification"
)

st.sidebar.write(
    "Classes: 5"
)

st.sidebar.write(
    "Input Size: 224 × 224"
)

st.sidebar.write(
    "Validation Accuracy: ~92%"
)


# =========================================================
# UPLOAD VIDEO
# =========================================================

uploaded_video = st.file_uploader(
    "🎥 Upload a driver video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)


# =========================================================
# MAIN
# =========================================================

if uploaded_video is not None:

    # =====================================================
    # SAVE INPUT VIDEO
    # =====================================================

    extension = os.path.splitext(
        uploaded_video.name
    )[1].lower()

    if extension not in [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    ]:
        extension = ".mp4"


    input_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension
    )

    input_file.write(
        uploaded_video.getbuffer()
    )

    input_path = input_file.name

    input_file.close()


    # =====================================================
    # ORIGINAL VIDEO
    # =====================================================

    st.markdown("---")

    st.markdown(
        """
        <div class="video-title">
            🎥 Original Video
        </div>
        """,
        unsafe_allow_html=True
    )

    st.video(
        input_path
    )


    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    st.markdown("")

    analyze_button = st.button(
        "🚀 Analyze Driver Behavior",
        use_container_width=True
    )


    # =====================================================
    # ANALYSIS
    # =====================================================

    if analyze_button:

        # -------------------------------------------------
        # Temporary output
        # -------------------------------------------------

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_path = output_file.name

        output_file.close()


        # -------------------------------------------------
        # Open video
        # -------------------------------------------------

        cap = cv2.VideoCapture(
            input_path
        )


        if not cap.isOpened():

            st.error(
                "❌ Could not open the video."
            )

            st.stop()


        # -------------------------------------------------
        # Video properties
        # -------------------------------------------------

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 30


        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        # -------------------------------------------------
        # Video writer
        # -------------------------------------------------

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )


        if not out.isOpened():

            cap.release()

            st.error(
                "❌ Could not create output video."
            )

            st.stop()


        # -------------------------------------------------
        # Smoothing
        # -------------------------------------------------

        prediction_history = deque(
            maxlen=SMOOTHING_WINDOW
        )


        # Last prediction

        last_prediction = np.ones(
            len(CLASS_NAMES),
            dtype=np.float32
        ) / len(CLASS_NAMES)


        # -------------------------------------------------
        # Statistics
        # -------------------------------------------------

        behavior_counts = {
            name: 0
            for name in CLASS_NAMES
        }


        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        progress = st.progress(0)

        status = st.empty()


        # =================================================
        # PROCESS FRAMES
        # =================================================

        frame_number = 0


        while True:

            ret, frame = cap.read()

            if not ret:
                break


            frame_number += 1


            # =================================================
            # PREDICTION
            # =================================================

            if (
                frame_number == 1
                or frame_number % PREDICT_EVERY_N_FRAMES == 0
            ):

                # BGR -> RGB

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )


                # Resize

                image = cv2.resize(
                    rgb,
                    (IMG_SIZE, IMG_SIZE),
                    interpolation=cv2.INTER_AREA
                )


                # IMPORTANT:
                # Do NOT divide by 255.
                # EfficientNet preprocessing is included
                # in the Keras model.

                image = image.astype(
                    np.float32
                )


                # Add batch dimension

                image = np.expand_dims(
                    image,
                    axis=0
                )


                # Force exact shape

                image = image.reshape(
                    1,
                    IMG_SIZE,
                    IMG_SIZE,
                    3
                )


                # Prediction

                prediction = model.predict(
                    image,
                    verbose=0
                )


                # Convert safely

                prediction = np.asarray(
                    prediction,
                    dtype=np.float32
                ).reshape(-1)


                # Safety check

                if len(prediction) != len(
                    CLASS_NAMES
                ):

                    cap.release()
                    out.release()

                    st.error(
                        "❌ Model output does not match "
                        "the five class names."
                    )

                    st.stop()


                # Normalize probabilities

                prediction_sum = np.sum(
                    prediction
                )

                if prediction_sum > 0:

                    prediction = (
                        prediction /
                        prediction_sum
                    )


                # Save prediction

                last_prediction = (
                    prediction.copy()
                )


                # Current class

                current_index = int(
                    np.argmax(
                        prediction
                    )
                )


                prediction_history.append(
                    current_index
                )


            # =================================================
            # TEMPORAL SMOOTHING
            # =================================================

            if len(prediction_history) > 0:

                counter = Counter(
                    prediction_history
                )

                smoothed_index = (
                    counter.most_common(1)[0][0]
                )

            else:

                smoothed_index = int(
                    np.argmax(
                        last_prediction
                    )
                )


            # =================================================
            # BEHAVIOR
            # =================================================

            behavior = CLASS_NAMES[
                smoothed_index
            ]

            display_behavior = behavior.replace(
                "_",
                " "
            ).upper()


            # =================================================
            # CONFIDENCE
            # =================================================

            confidence = float(
                last_prediction[
                    smoothed_index
                ]
            )

            confidence_percent = (
                confidence * 100
            )


            # =================================================
            # STATISTICS
            # =================================================

            behavior_counts[
                behavior
            ] += 1


            # =================================================
            # DRAW OVERLAY
            # =================================================

            overlay = frame.copy()


            # Information box

            box_width = min(
                580,
                width - 20
            )

            box_height = min(
                145,
                height - 20
            )


            cv2.rectangle(
                overlay,
                (20, 20),
                (
                    box_width,
                    box_height
                ),
                (15, 15, 15),
                -1
            )


            # Transparency

            frame = cv2.addWeighted(
                overlay,
                0.78,
                frame,
                0.22,
                0
            )


            # =================================================
            # TITLE
            # =================================================

            cv2.putText(
                frame,
                "DRIVER BEHAVIOR",
                (40, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


            # =================================================
            # BEHAVIOR
            # =================================================

            cv2.putText(
                frame,
                display_behavior,
                (40, 93),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 105, 180),
                2,
                cv2.LINE_AA
            )


            # =================================================
            # CONFIDENCE
            # =================================================

            cv2.putText(
                frame,
                f"Confidence: "
                f"{confidence_percent:.1f}%",
                (40, 123),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                1,
                cv2.LINE_AA
            )


            # =================================================
            # WRITE FRAME
            # =================================================

            out.write(
                frame
            )


            # =================================================
            # PROGRESS
            # =================================================

            if total_frames > 0:

                progress_value = (
                    frame_number /
                    total_frames
                )

                progress.progress(
                    min(
                        progress_value,
                        1.0
                    )
                )

                status.text(
                    f"Processing frame "
                    f"{frame_number:,} / "
                    f"{total_frames:,}"
                )


        # =====================================================
        # RELEASE VIDEO
        # =====================================================

        cap.release()
        out.release()


        # =====================================================
        # CONVERT TO H264
        # =====================================================

        status.text(
            "🔄 Converting video for browser playback..."
        )


        h264_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        h264_output_path = h264_file.name

        h264_file.close()


        try:

            import imageio_ffmpeg

            ffmpeg_exe = (
                imageio_ffmpeg.get_ffmpeg_exe()
            )


            subprocess.run(
                [
                    ffmpeg_exe,
                    "-y",
                    "-i",
                    output_path,

                    # H.264
                    "-c:v",
                    "libx264",

                    # Fast encoding
                    "-preset",
                    "fast",

                    # Quality
                    "-crf",
                    "23",

                    # Browser compatible pixel format
                    "-pix_fmt",
                    "yuv420p",

                    # Better browser streaming
                    "-movflags",
                    "+faststart",

                    h264_output_path
                ],
                check=True
            )


            # Use browser-friendly file

            output_path = (
                h264_output_path
            )


        except Exception as e:

            st.warning(
                "⚠️ H.264 conversion failed. "
                "The original MP4 will be used."
            )

            st.code(
                str(e)
            )


        # =====================================================
        # COMPLETE
        # =====================================================

        progress.progress(1.0)

        status.success(
            "✅ Driver behavior analysis completed!"
        )


        # =====================================================
        # SIDE-BY-SIDE VIDEO COMPARISON
        # =====================================================

        st.markdown("---")

        st.markdown(
            """
            <h2 style="
                text-align:center;
                margin-bottom:25px;
            ">
                🎬 Driver Behavior Analysis
            </h2>
            """,
            unsafe_allow_html=True
        )


        # EXACTLY TWO EQUAL COLUMNS

        col1, col2 = st.columns(
            [1, 1],
            gap="large"
        )


        # =====================================================
        # LEFT VIDEO
        # =====================================================

        with col1:

            st.markdown(
                """
                <div style="
                    text-align:center;
                    font-size:22px;
                    font-weight:bold;
                    margin-bottom:10px;
                ">
                    🎥 Original Video
                </div>
                """,
                unsafe_allow_html=True
            )


            st.video(
                input_path
            )


        # =====================================================
        # RIGHT VIDEO
        # =====================================================

        with col2:

            st.markdown(
                """
                <div style="
                    text-align:center;
                    font-size:22px;
                    font-weight:bold;
                    margin-bottom:10px;
                ">
                    🤖 AI Predicted Video
                </div>
                """,
                unsafe_allow_html=True
            )


            st.video(
                output_path
            )


        # =====================================================
        # STATISTICS
        # =====================================================

        st.markdown("---")

        st.markdown(
            """
            <h2 style="
                text-align:center;
            ">
                📊 Detected Behaviors
            </h2>
            """,
            unsafe_allow_html=True
        )


        total_predictions = sum(
            behavior_counts.values()
        )


        if total_predictions > 0:

            metric_cols = st.columns(
                len(CLASS_NAMES)
            )


            for i, class_name in enumerate(
                CLASS_NAMES
            ):

                count = behavior_counts[
                    class_name
                ]


                percentage = (
                    count /
                    total_predictions
                ) * 100


                with metric_cols[i]:

                    st.metric(
                        label=class_name.replace(
                            "_",
                            " "
                        ).title(),
                        value=f"{percentage:.1f}%"
                    )


        # =====================================================
        # DOWNLOAD
        # =====================================================

        st.markdown("---")


        with open(
            output_path,
            "rb"
        ) as video_file:

            st.download_button(
                label="⬇️ Download Predicted Video",
                data=video_file,
                file_name="driver_behavior_analysis.mp4",
                mime="video/mp4",
                use_container_width=True
            )