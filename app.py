import streamlit as st
from ultralytics import YOLO
import requests
from PIL import Image
import io

# Load the YOLO model
model = YOLO(r"D:\Artificial Intelligence and Machine Learning\Projects\AICTE\Techsaksham Edunet\Brain Tumor Detection using YOLO\runs\detect\train2\weights\best.pt")  # Replace with your model path

# Streamlit App UI
st.title("Brain Tumor Detection")
st.header("Upload an image or provide an image URL for prediction")

# File uploader or URL input
image_url = st.text_input("Enter an image URL:")

uploaded_image = st.file_uploader("Or upload an image directly", type=["jpg", "png", "jpeg"])

if uploaded_image:
    # If image is uploaded, process it
    image = Image.open(uploaded_image)
    # Run the prediction
    results = model.predict(image)

elif image_url:
    # If URL is provided, fetch the image
    try:
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        # Run the prediction
        results = model.predict(image)

    except Exception as e:
        st.error(f"Error loading image from URL: {e}")

# If results are available
if 'results' in locals() and results:
    # Since `results` is a list, access the first element (the result object)
    result = results[0]  # Get the first result (for single image prediction)

    # Plot the results (draw bounding boxes, labels, and confidence)
    img = result.plot()  # This returns the image with annotations

    # Create two columns for side-by-side layout of images
    col1, col2 = st.columns(2)

    # Show the uploaded image in the first column
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Show the predicted image in the second column
    with col2:
        st.image(img, caption="Predicted Image", use_container_width=True)

    # Access the detected boxes directly
    boxes = result.boxes  # This will give the Boxes object containing detected boxes
    detected_objects = []

    if boxes is not None and len(boxes) > 0:
        for i, box in enumerate(boxes):
            # Extract confidence and class ID
            confidence = box.conf.item()  # Confidence score
            class_id = int(box.cls.item())  # Class ID
            class_name = result.names[class_id]  # Get the class name from the model's names mapping

            # Only process if confidence is above a threshold (e.g., 0.5)
            if confidence >= 0.5:
                detected_objects.append(f"**Object {i + 1}:** {class_name} with confidence {confidence * 100:.2f}%")
            else:
                detected_objects.append(f"**Object {i + 1}:** Unknown with confidence {confidence * 100:.2f}%")
    else:
        detected_objects.append("No tumor detected with high confidence.")

    # Show detected objects and explanation in the full width below the images
    st.write("### Detected Objects:")
    if detected_objects:
        for obj in detected_objects:
            st.write(obj)
    else:
        st.write("No objects detected.")

    # Display Results Explanation in full width
    st.write("### Results Explanation:")
    if detected_objects and any("positive" in obj.lower() for obj in detected_objects):
        st.write(
            """
            The model has identified potential abnormal regions in the brain image that may indicate the presence of a tumor. 
            These regions are highlighted with bounding boxes, and each detection is accompanied by a confidence score, 
            which represents the model's certainty about the finding.

            **Clinical Interpretation:**
            - The detected regions could be areas of concern, such as tumors or other abnormalities.
            - A higher confidence score suggests a stronger likelihood of the presence of a tumor.
            - However, this is a preliminary analysis. Further diagnostic tests, such as MRI scans, CT scans, or a biopsy, 
            are recommended to confirm the findings and determine the nature of the abnormality.

            **Next Steps:**
            1. Consult a neurologist or oncologist for a detailed evaluation.
            2. Perform additional imaging tests for a comprehensive diagnosis.
            3. Discuss treatment options if a tumor is confirmed.
            """
        )
    else:
        st.write(
            """
            The model did not detect any tumor regions with high confidence in the brain image. This could mean one of the following:

            **Clinical Interpretation:**
            - No tumor is present in the analyzed image.
            - The tumor, if present, might be too small or not clearly visible in the provided image.
            - The image quality or resolution may not be sufficient for the model to detect subtle abnormalities.

            **Next Steps:**
            1. If symptoms persist, consult a neurologist for further evaluation.
            2. Consider advanced imaging techniques (e.g., MRI with contrast) for a more detailed analysis.
            3. Regular follow-ups and monitoring are recommended to ensure early detection of any potential issues.
            """
        )