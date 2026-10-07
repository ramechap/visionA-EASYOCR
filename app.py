import json
import io

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance
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
    value=0.20,
    step=0.05,
)


upscale_factor = st.sidebar.selectbox(
    "Image upscaling",
    options=[1, 2, 3],
    index=2,
    help=(
        "Upscaling can improve recognition of small text. "
        "3x is useful for CVs and documents."
    ),
)


use_gpu = st.sidebar.checkbox(
    "Use GPU",
    value=False,
    help=(
        "Enable only if your hosting environment has "
        "a compatible GPU."
    ),
)


# =========================================================
# LOAD EASYOCR MODEL
# =========================================================

@st.cache_resource
def load_reader(languages, gpu):
    """
    Load EasyOCR once and cache it.
    """

    return easyocr.Reader(
        list(languages),
        gpu=gpu,
        verbose=False,
    )


with st.spinner("Loading EasyOCR model..."):

    try:

        reader = load_reader(
            tuple(language_codes),
            use_gpu,
        )

    except Exception as e:

        st.error(
            "Could not load EasyOCR.\n\n"
            f"Error: {e}"
        )

        st.stop()


# =========================================================
# IMAGE PREPARATION
# =========================================================

def prepare_image(uploaded_file):
    """
    Open uploaded image and convert it to RGB.
    """

    image = Image.open(uploaded_file)

    image.load()

    return image.convert("RGB")


def preprocess_image(image, scale=3):
    """
    Improve small-text recognition using PIL only.

    No cv2 is required.
    """

    if scale > 1:

        new_width = image.width * scale
        new_height = image.height * scale

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS,
        )

    # Slight contrast improvement
    image = ImageEnhance.Contrast(
        image
    ).enhance(1.15)

    # Slight sharpness improvement
    image = ImageEnhance.Sharpness(
        image
    ).enhance(1.25)

    return image


# =========================================================
# DRAW OCR BOXES
# =========================================================

def draw_ocr_boxes(image, results, scale=1):
    """
    Draw EasyOCR bounding boxes on the original image.

    Coordinates are converted back to the original
    image size when upscaling was used.
    """

    output = image.copy()

    draw = ImageDraw.Draw(output)

    for item in results:

        box = item["box"]
        text = item["text"]
        confidence = item["confidence"]

        # Convert processed-image coordinates
        # back to original-image coordinates.
        points = []

        for point in box:

            x = int(point[0] / scale)
            y = int(point[1] / scale)

            points.append(
                (x, y)
            )

        if len(points) < 4:
            continue

        # Draw bounding box
        draw.line(
            points + [points[0]],
            width=2,
            fill="red",
        )

        # Find top-left position
        x = min(
            point[0]
            for point in points
        )

        y = min(
            point[1]
            for point in points
        )

        label = (
            f"{text} "
            f"({confidence:.2f})"
        )

        # Draw label
        draw.text(
            (
                x,
                max(0, y - 18),
            ),
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

    The returned data contains only normal Python
    types so it can safely be converted to JSON.
    """

    # -----------------------------------------------------
    # PREPROCESS / UPSCALE
    # -----------------------------------------------------

    processed_image = preprocess_image(
        image,
        scale=upscale_factor,
    )

    # -----------------------------------------------------
    # PIL -> NumPy
    # -----------------------------------------------------

    image_array = np.asarray(
        processed_image
    )

    # -----------------------------------------------------
    # EASY OCR
    # -----------------------------------------------------

    results = reader.readtext(
        image_array,
        detail=1,
        paragraph=False,

        # Helps with smaller text
        mag_ratio=1.5,
        canvas_size=4000,

        # More sensitive text detection
        text_threshold=0.4,
        low_text=0.2,
        link_threshold=0.2,

        # Text grouping
        width_ths=0.7,
        height_ths=0.5,
    )

    extracted_results = []

    # -----------------------------------------------------
    # CONVERT EVERYTHING TO PYTHON TYPES
    # -----------------------------------------------------

    for result in results:

        if len(result) < 3:
            continue

        raw_box = result[0]
        raw_text = result[1]
        raw_confidence = result[2]

        text = str(
            raw_text
        ).strip()

        confidence = float(
            raw_confidence
        )

        if not text:
            continue

        if confidence < confidence_threshold:
            continue

        # -------------------------------------------------
        # IMPORTANT:
        # Convert NumPy int32/int64 values into normal
        # Python int values.
        # -------------------------------------------------

        safe_box = []

        for point in raw_box:

            x = int(
                point[0]
            )

            y = int(
                point[1]
            )

            safe_box.append(
                [x, y]
            )

        extracted_results.append(
            {
                "box": safe_box,
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
        "Upload an image containing text "
        "to begin OCR."
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
    width="stretch",
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

            # Store results in session state
            st.session_state[
                "ocr_results"
            ] = results

            # Store image scale
            st.session_state[
                "ocr_scale"
            ] = upscale_factor

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

    result_scale = st.session_state.get(
        "ocr_scale",
        1,
    )

    st.divider()

    # -----------------------------------------------------
    # NO TEXT
    # -----------------------------------------------------

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
    # CREATE BOXED IMAGE
    # =====================================================

    boxed_image = draw_ocr_boxes(
        image,
        results,
        scale=result_scale,
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
            height=400,
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

        st.image(
            boxed_image,
            caption="Detected text regions",
            width="stretch",
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


        # -------------------------------------------------
        # TEXT DOWNLOAD
        # -------------------------------------------------

        st.download_button(
            label="📄 Download TXT",
            data=extracted_text,
            file_name="ocr_result.txt",
            mime="text/plain",
        )


        # -------------------------------------------------
        # JSON DOWNLOAD
        # -------------------------------------------------

        # Create a completely JSON-safe copy.
        #
        # This is intentionally separate from `results`
        # so NumPy values can never reach json.dumps().

        json_safe_results = []

        for item in results:

            safe_box = []

            for point in item["box"]:

                safe_box.append(
                    [
                        int(point[0]),
                        int(point[1]),
                    ]
                )

            json_safe_results.append(
                {
                    "box": safe_box,
                    "text": str(
                        item["text"]
                    ),
                    "confidence": float(
                        item["confidence"]
                    ),
                }
            )


        json_data = json.dumps(
            json_safe_results,
            ensure_ascii=False,
            indent=2,
        )


        st.download_button(
            label="📋 Download JSON",
            data=json_data,
            file_name="ocr_result.json",
            mime="application/json",
        )


        # -------------------------------------------------
        # IMAGE DOWNLOAD
        # -------------------------------------------------

        image_buffer = io.BytesIO()

        boxed_image.save(
            image_buffer,
            format="PNG",
        )

        image_buffer.seek(0)


        st.download_button(
            label="🖼️ Download Image With Boxes",
            data=image_buffer.getvalue(),
            file_name="ocr_boxes.png",
            mime="image/png",
        )


        # -------------------------------------------------
        # OCR SUMMARY
        # -------------------------------------------------

        st.divider()

        st.write(
            f"**Detected text regions:** "
            f"{len(results)}"
        )

        st.write(
            f"**Languages:** "
            f"{', '.join(selected_languages)}"
        )

        st.write(
            f"**Minimum confidence:** "
            f"{confidence_threshold:.0%}"
        )

        st.write(
            f"**Image upscaling:** "
            f"{upscale_factor}×"
        )
