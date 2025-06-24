# Glaucoma Detection System using YOLOv8 & Flask

**Introduction**

In the field of ophthalmology, early detection of Glaucoma—a leading cause of irreversible blindness—is critical. However, manual examination of fundus images and videos to determine the Cup-to-Disc Ratio (CDR) is time-consuming and prone to human error. To address this challenge, we present an AI-powered solution: **Glaucoma Detection System using YOLOv8 & Flask**. This project leverages the power of Deep Learning to analyze fundus images and videos, accurately compute the CDR, and predict the risk of Glaucoma in real-time. Built with Flask and deployed on Hugging Face Spaces, this user-friendly system aims to assist ophthalmologists and researchers in efficient, automated diagnosis.

<br />

## Table of Contents

1. [Key Technologies and Skills](#key-technologies-and-skills)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [Features](#features)  
5. [Contributing](#contributing)  
6. [License](#license)  
7. [Contact](#contact)

<br />

## Key Technologies and Skills

- Python  
- YOLOv8 (Ultralytics)  
- Computer Vision  
- OpenCV  
- Kalman Filter  
- Flask  
- HTML5 + JavaScript  
- Hugging Face Spaces

<br />

## Installation

To run this project, you need to install the following packages:

```bash
pip install flask
pip install ultralytics
pip install opencv-python-headless
pip install numpy
pip install torch
pip install pykalman

Note: If you're using a lightweight machine or Hugging Face Space, it's recommended to use opencv-python-headless instead of opencv-python.
<br />
Usage

To use this project, follow these steps:

Clone the repository: git clone https://github.com/souviksenapati/Glaucoma-Detection-System

Navigate to the project folder:

cd Glaucoma-Detection-System

Install the required packages:

pip install -r requirements.txt

Run the Flask app:

python app.py

Access the app in your browser at:

    http://127.0.0.1:5000

<br />
Features
Fundus Image & Video Analysis:

    Accepts input in the form of .jpg, .png (images) or .mp4, .avi (videos).

    Automatically detects optic disc and optic cup regions using YOLOv8.

    Calculates the Cup-to-Disc Ratio (CDR) for each frame (in case of video).

    Applies Kalman Filtering for smoothing predictions across frames.

    Generates a final CDR and provides a Glaucoma Risk Diagnosis:

        CDR < 0.4 → Healthy

        CDR 0.4 - 0.5 → Risk Detected (percentage shown)

        CDR ≥ 0.5 → Glaucoma Detected

Web Interface:

    Built using Flask + HTML5 + Bootstrap.

    Upload form with loader animation and progress bars.

    Dynamic result display with color-coded tags (Healthy / Risk / Glaucoma).

    Optimized for deployment on Hugging Face Spaces.

Model:

    Trained on real-world annotated fundus datasets.

    Deployed using CPU-safe PyTorch + YOLOv8 pipeline.

    Smoothened using a Kalman Filter to reduce prediction noise in videos.

<br />
🚀 Live Demo

Try the app live on Hugging Face Spaces:
👉 https://huggingface.co/spaces/souviksenapati/Glaucoma-Detection-System
<br />
Contributing

Contributions are welcome!
Feel free to fork the repository, suggest enhancements, open issues, or submit pull requests. Whether it's code optimization, UI improvement, or feature ideas — all help is appreciated!
<br />
License

This project is licensed under the MIT License.
See the LICENSE file for more information.
<br />
Contact

📧 Email: souviksenapati85@gmail.com

If you have any questions, suggestions, or collaboration opportunities, feel free to reach out!
