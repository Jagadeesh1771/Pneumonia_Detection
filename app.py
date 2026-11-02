
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="PneumoVision AI",
    page_icon="🫁",
    layout="wide"
)

# CSS styling
st.markdown("""
<style>
.main {
    background-color: #0b1020;
}
[data-testid="stSidebar"] {
    background-color: #101a30;
}
h1, h2, h3 {
    color: #60a5fa;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    width: 100%;
    height: 45px;
}
</style>
""", unsafe_allow_html=True)


# Load CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "cnn_pneumonia_model.keras"
    )

cnn_model = load_model()


# Sidebar navigation - ONLY TWO PAGES
st.sidebar.title("🫁 PneumoVision AI")
st.sidebar.caption("AI-Powered Chest X-Ray Analysis")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Upload X-Ray"]
)

st.sidebar.divider()

st.sidebar.info(
    "Educational and research purposes only. "
    "Not for clinical use."
)


# Prediction function
def predict_image(image):

    resized = image.resize((224, 224))

    arr = np.array(resized, dtype=np.float32)
    arr = arr / 255.0

    arr = np.expand_dims(arr, axis=0)

    prediction = cnn_model.predict(
        arr, verbose=0
    )[0][0]

    return float(prediction), arr


# Grad-CAM function
def generate_gradcam(img_array):

    gradcam_input = tf.keras.Input(
        shape=(224, 224, 3)
    )

    x = gradcam_input
    last_conv_output = None

    for layer in cnn_model.layers:

        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        x = layer(x)

        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_output = x

    grad_model = tf.keras.Model(
        inputs=gradcam_input,
        outputs=[last_conv_output, x]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array,
            training=False
        )

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()


# Create overlay
def make_overlay(image, heatmap):

    original = np.array(
        image.resize((224, 224)).convert("RGB")
    )

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        (224, 224)
    ).numpy().squeeze()

    colormap = plt.get_cmap("jet")

    colored = (
        colormap(heatmap_resized)[:, :, :3] * 255
    ).astype(np.uint8)

    overlay = (
        0.6 * original + 0.4 * colored
    ).clip(0, 255).astype(np.uint8)

    return colored, overlay


# ==================================
# PAGE 1 - HOME
# ==================================

if page == "Home":

    st.title("🫁 PneumoVision AI")

    st.subheader(
        "AI-Powered Pneumonia Detection from Chest X-Rays"
    )

    st.write(
        "An intelligent deep learning application "
        "designed to classify chest X-ray images "
        "into NORMAL and PNEUMONIA categories."
    )

    st.divider()

    st.header("About Our Project")

    st.write("""
    PneumoVision AI is a deep learning-based project
    that uses Convolutional Neural Networks to analyze
    chest X-ray images.

    The system uses image preprocessing, deep learning
    classification and Grad-CAM explainability to
    demonstrate how AI can analyze medical images.
    """)

    st.header("Project Objectives")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        - Automate chest X-ray image classification.
        - Identify NORMAL and PNEUMONIA patterns.
        - Apply deep learning techniques.
        """)

    with c2:
        st.markdown("""
        - Visualize model activation using Grad-CAM.
        - Evaluate model performance.
        - Develop an interactive web application.
        """)

    st.divider()

    st.header("Project Workflow")

    st.markdown("""
    **1. Image Upload** → **2. Image Preprocessing**
    → **3. CNN Classification** → **4. Prediction**
    → **5. Grad-CAM Visualization**
    """)

    st.header("Technologies Used")

    t1, t2, t3, t4 = st.columns(4)

    t1.info("Python")
    t2.info("TensorFlow")
    t3.info("CNN")
    t4.info("Streamlit")

    t1, t2, t3, t4 = st.columns(4)

    t1.info("NumPy")
    t2.info("Pandas")
    t3.info("Matplotlib")
    t4.info("Grad-CAM")

    st.divider()

    st.header("Project Features")

    f1, f2, f3 = st.columns(3)

    f1.markdown("""
    **Image Classification**

    Chest X-ray classification into two classes.
    """)

    f2.markdown("""
    **Deep Learning**

    Custom CNN trained using chest X-ray images.
    """)

    f3.markdown("""
    **Explainable AI**

    Grad-CAM heatmap visualization.
    """)

    st.divider()

    st.info(
        "Use the Upload X-Ray page in the sidebar "
        "to upload an image and view the experimental prediction."
    )

    st.warning(
        "Disclaimer: This project is for educational "
        "and research purposes only. It is not a "
        "medical diagnostic system."
    )

    st.caption(
        "PneumoVision AI | MSc AI & Data Science Project"
    )


# ==================================
# PAGE 2 - UPLOAD X-RAY
# ==================================

elif page == "Upload X-Ray":

    st.title("Chest X-Ray Analysis")

    st.write(
        "Upload your chest X-ray image to analyze it "
        "using the trained Custom CNN model."
    )

    uploaded_file = st.file_uploader(
        "Upload Chest X-Ray Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Original Chest X-Ray")

        st.image(
            image,
            use_container_width=True
        )

        if st.button("Analyze X-Ray"):

            with st.spinner("Analyzing image..."):

                probability, img_array = predict_image(image)

                if probability >= 0.5:
                    result = "PNEUMONIA"
                    confidence = probability
                else:
                    result = "NORMAL"
                    confidence = 1 - probability

                heatmap = generate_gradcam(img_array)

                colored, overlay = make_overlay(
                    image, heatmap
                )

                st.session_state["result"] = result
                st.session_state["probability"] = probability
                st.session_state["confidence"] = confidence
                st.session_state["heatmap"] = colored
                st.session_state["overlay"] = overlay

        if "result" in st.session_state:

            st.divider()

            st.header("AI Prediction")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Predicted Class",
                st.session_state["result"]
            )

            c2.metric(
                "Confidence",
                f"{st.session_state['confidence'] * 100:.2f}%"
            )

            c3.metric(
                "Pneumonia Probability",
                f"{st.session_state['probability'] * 100:.2f}%"
            )

            st.progress(
                st.session_state["confidence"]
            )

            st.divider()

            st.header("Explainable AI - Grad-CAM")

            g1, g2 = st.columns(2)

            with g1:
                st.image(
                    st.session_state["heatmap"],
                    caption="Grad-CAM Heatmap",
                    use_container_width=True
                )

            with g2:
                st.image(
                    st.session_state["overlay"],
                    caption="Grad-CAM Overlay",
                    use_container_width=True
                )

            st.warning(
                "Grad-CAM shows image regions influencing "
                "the model. It does not confirm pneumonia "
                "or provide a medical diagnosis."
            )

    else:

        st.info(
            "Please upload a chest X-ray image to begin analysis."
        )

    st.divider()

    st.caption(
        "PneumoVision AI | Educational Use Only"
    )
