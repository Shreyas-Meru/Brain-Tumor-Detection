# Brain Tumor Detection using YOLO

## Overview
This project implements a **Brain Tumor Detection System** using **YOLO (You Only Look Once)** for real-time object detection in medical images. It utilizes **deep learning and computer vision** techniques to detect brain tumors in MRI scans with high accuracy. A **Streamlit-based web application** is provided for easy interaction, allowing users to upload images or provide image URLs for detection.

## Features
- **Real-time brain tumor detection** using YOLO.
- **User-friendly Streamlit interface** for uploading or entering image URLs.
- **Displays results with bounding boxes and confidence scores**.
- **Fast and efficient model inference**.

## Snapshots

### **1. Uploading an Image for Detection**
![image](https://github.com/user-attachments/assets/da0320c9-d29a-40b9-885a-dcdd797f1dff)

*Users can upload an MRI scan or enter an image URL.*

### **2. Detection Output**
![image](https://github.com/user-attachments/assets/5952c5f1-ef78-4385-a2ac-7fb2cd0d037c)

*The left side shows the uploaded brain scan, while the right side displays the detected tumor region with a bounding box.*

### **3. Detection Results Summary**
![image](https://github.com/user-attachments/assets/5dc2b22f-a694-4b29-a71f-0a9cdde670ba)

*A summary of the detected tumor, including confidence scores.*

## Installation & Setup
### **Prerequisites**
- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Streamlit
- PIL
- Requests

### **Installation Steps**
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/brain-tumor-detection-yolo.git
   cd brain-tumor-detection-yolo
   ```
2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Download the trained YOLO model weights**
   - Place `best.pt` inside the `weights/` directory.
   
4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the **Streamlit web interface** in your browser.
2. Upload a brain MRI scan or provide an image URL.
3. Click **"Detect Tumor"** to process the image.
4. View results with **bounding boxes and confidence scores**.

## Model Training
To train YOLO on a custom dataset:
```bash
yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=50
```

## Future Enhancements
- **Integration with segmentation models** (e.g., U-Net) for precise tumor boundary detection.
- **Multi-modal analysis** using MRI & CT scans.
- **Deploy as a cloud-based API** for real-world clinical usage.

## Contributing
Feel free to **fork, improve, and contribute**! Submit a pull request with your changes.

## License
This project is open-source under the **Shreyas Meru License**.

## Contact
For queries, reach out at LinkedIn: `@ShreyasMeru` or visit [GitHub Profile](https://github.com/shreyas-meru).
