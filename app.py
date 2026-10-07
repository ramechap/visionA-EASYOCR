import json
import io

import streamlit as st
from PIL import Image, ImageDraw
import easyocr


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EasyOCR Text Scanner",
    page_icon="🔤",
    layout="wide",
)


st.title("🔤 EasyOCR Text Scanner")
st.write(
    "Upload an image and extract readable text using EasyOCR."
)


# =========================================================
# LANGUAGE SETTINGS
# =========================================================

LANGUAGES = {
    "English": "en",
    "Nepali": "ne",
    "Hindi": "hi",
}


selected_languages = st.sidebar.multiselect(
    "OCR Languages",
    options=list(LANGUAGES.keys()),
    default=["English"],
)


if not selected_languages:
    st.warning("Please select at least one OCR language.")
    st.stop()


language_codes = [
    LANGUAGES[language]
    for language in selected_languages
]


# =========================================================
# OCR SETTINGS
# =========================================================

confidence_threshold = st.sidebar.slider(
    "Minimum confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.05,
)


use_gpu = st.sidebar.checkbox(
    "Use GPU",
    value=False,
    help="Enable this only if your hosting environment has a compatible GPU.",
)


# =========================================================
# LOAD EASYOCR MODEL
# =========================================================

@st.cache_resource
def load_reader(languages, gpu):
    """
    Load EasyOCR once and cache it.

    This prevents the OCR model from being loaded
    every time the user clicks the OCR button.
    """

    return easyocr.Reader(
        languages,
        gpu=gpu,
        verbose=False,
    )


with st.spinner("Loading EasyOCR model..."):
    try:
        reader = load_reader(
            language_codes,
            use_gpu,
        )

    except Exception as e:
        st.error(
            "Could not load EasyOCR.\n\n"
            f"Error: {e}"
        )
        st.stop()


# =========================================================
# IMAGE PROCESSING
# =========================================================

def prepare_image(uploaded_file):
    """
    Open uploaded image and convert it to RGB.
    """

    image = Image.open(uploaded_file)

    image.load()

    return image.convert("RGB")


def draw_ocr_boxes(image, results):
    """
    Draw EasyOCR bounding boxes and confidence
    values on the image.
    """

    output = image.copy()

    draw = ImageDraw.Draw(output)

    for item in results:

        box = item["box"]
        text = item["text"]
        confidence = item["confidence"]

        points = [
            tuple(point)
            for point in box
        ]

        draw.line(
            points + [points[0]],
            width=3,
            fill="red",
        )

        x = int(min(point[0] for point in points))
        y = int(min(point[1] for point in points))

        label = (
            f"{text} "
            f"({confidence:.2f})"
        )

        draw.text(
            (x, max(0, y - 20)),
            label,
            fill="red",
        )

    return output


# =========================================================
# OCR FUNCTION
# =========================================================

def perform_ocr(image):
    """
    Run EasyOCR on the image.
    """

    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
    )

    extracted_results = []

    for result in results:

        box = result[0]
        text = result[1]
        confidence = float(result[2])

        if confidence < confidence_threshold:
            continue

        extracted_results.append(
            {
                "box": box,
                "text": text,
                "confidence": confidence,
            }
        )

    return extracted_results


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)


if uploaded_file is None:

    st.info(
        "Upload an image containing text to begin OCR."
    )

    st.stop()


# =========================================================
# OPEN IMAGE
# =========================================================

try:

    image = prepare_image(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Could not open the image: {e}"
    )

    st.stop()


# =========================================================
# SHOW ORIGINAL IMAGE
# =========================================================

st.subheader("📷 Original Image")

st.image(
    image,
    caption=uploaded_file.name,
    use_container_width=True,
)


# =========================================================
# RUN OCR
# =========================================================

if st.button(
    "🔤 Extract Text",
    type="primary",
):

    with st.spinner(
        "Detecting and recognizing text..."
    ):

        try:

            results = perform_ocr(
                image
            )

            st.session_state[
                "ocr_results"
            ] = results

        except Exception as e:

            st.error(
                f"OCR failed: {e}"
            )

            st.stop()


# =========================================================
# DISPLAY RESULTS
# =========================================================

if "ocr_results" in st.session_state:

    results = st.session_state[
        "ocr_results"
    ]

    st.divider()

    if not results:

        st.warning(
            "No readable text was detected."
        )

        st.stop()


    # =====================================================
    # EXTRACT TEXT
    # =====================================================

    extracted_text = "\n".join(
        item["text"]
        for item in results
    )


    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3 = st.tabs(
        [
            "📝 Text",
            "📦 Detected Text Boxes",
            "📥 Downloads",
        ]
    )


    # =====================================================
    # TEXT TAB
    # =====================================================

    with tab1:

        st.subheader(
            "📝 Extracted Text"
        )

        st.text_area(
            "OCR Result",
            value=extracted_text,
            height=350,
        )

        st.metric(
            "Text regions detected",
            len(results),
        )


    # =====================================================
    # BOUNDING BOX TAB
    # =====================================================

    with tab2:

        st.subheader(
            "📦 Text Detection"
        )

        boxed_image = draw_ocr_boxes(
            image,
            results,
        )

        st.image(
            boxed_image,
            caption="Detected text regions",
            use_container_width=True,
        )


        st.subheader(
            "Detected Text"
        )

        for index, item in enumerate(
            results,
            start=1,
        ):

            st.write(
                f"**{index}.** "
                f"{item['text']}"
            )

            st.caption(
                f"Confidence: "
                f"{item['confidence']:.2%}"
            )


    # =====================================================
    # DOWNLOAD TAB
    # =====================================================

    with tab3:

        st.subheader(
            "📥 Download OCR Results"
        )


        # ---------------------------------------------
        # TEXT DOWNLOAD
        # ---------------------------------------------

        st.download_button(
            label="📄 Download TXT",
            data=extracted_text,
            file_name="ocr_result.txt",
            mime="text/plain",
        )


        # ---------------------------------------------
        # JSON DOWNLOAD
        # ---------------------------------------------

        json_data = json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        )


        st.download_button(
            label="📋 Download JSON",
            data=json_data,
            file_name="ocr_result.json",
            mime="application/json",
        )


        # ---------------------------------------------
        # IMAGE DOWNLOAD
        # ---------------------------------------------

        image_buffer = io.BytesIO()

        boxed_image.save(
            image_buffer,
            format="PNG",
        )


        st.download_button(
            label="🖼️ Download Image With Boxes",
            data=image_buffer.getvalue(),
            file_name="ocr_boxes.png",
            mime="image/png",
        )
