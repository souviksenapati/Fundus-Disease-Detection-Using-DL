import os
import cv2
import torch
import numpy as np
from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
from pykalman import KalmanFilter
import tempfile

app = Flask(__name__)

# ✅ Auto-select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔍 Using device: {device.upper()}")

# ✅ Load model
model_path = "./models/glucoma.pt"
model = YOLO(model_path).to(device)
model.model.float()

# ✅ Kalman Filter
def init_kalman():
    return KalmanFilter(initial_state_mean=0, n_dim_obs=1,
                        transition_matrices=[1],
                        observation_matrices=[1],
                        initial_state_covariance=1,
                        observation_covariance=1,
                        transition_covariance=0.01)

kalman_filter = init_kalman()
cdr_estimates = []

# ✅ Frame processing
def process_frame(frame):
    frame = frame.astype(np.float32)
    with torch.no_grad():
        results = model(frame, device=device)
    
    result = results[0]
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return None, None

    cup_box, disc_box = None, None
    cup_conf, disc_conf = 0, 0

    for box, cls, conf in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = box
        if cls == 0 and conf > cup_conf:
            cup_box = (x1, y1, x2, y2)
            cup_conf = conf
        elif cls == 1 and conf > disc_conf:
            disc_box = (x1, y1, x2, y2)
            disc_conf = conf

    if cup_conf < 0.3 or disc_conf < 0.3:
        return None, None

    # ✅ CDR Calculation
    cup_area = (cup_box[2] - cup_box[0]) * (cup_box[3] - cup_box[1])
    disc_area = (disc_box[2] - disc_box[0]) * (disc_box[3] - disc_box[1])
    cup_height = cup_box[3] - cup_box[1]
    disc_height = disc_box[3] - disc_box[1]
    cup_width = cup_box[2] - cup_box[0]
    disc_width = disc_box[2] - disc_box[0]

    cdr = ((cup_area / disc_area) + (cup_height / disc_height) + (cup_width / disc_width)) / 3
    return cdr, (cup_conf + disc_conf) / 2

# ✅ Full video processing
def calculate_cdr(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_cdr_values = []
    confidence_scores = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cdr, confidence = process_frame(frame)
        if cdr is not None and confidence is not None:
            frame_cdr_values.append(cdr)
            confidence_scores.append(confidence)

    cap.release()

    if len(frame_cdr_values) > 0:
        cdr_estimates.extend(frame_cdr_values)
        smoothed_cdr_values, _ = kalman_filter.filter(np.array(cdr_estimates))
        smoothed_cdr_values = smoothed_cdr_values.ravel()

        min_len = min(len(smoothed_cdr_values), len(confidence_scores))
        smoothed_cdr_values = smoothed_cdr_values[:min_len]
        confidence_scores = confidence_scores[:min_len]

        final_cdr = np.average(smoothed_cdr_values, weights=confidence_scores)
    else:
        final_cdr = None

    return final_cdr

# ✅ Diagnosis
def get_glaucoma_risk(final_cdr):
    if final_cdr is None:
        return "No valid detections", None

    if final_cdr < 0.4:
        return "No Glaucoma", 0
    
    elif 0.4 <= final_cdr < 0.5:
        risk_percentage = round(((final_cdr - 0.4) / 0.1) * 100, 1)
        return f"{risk_percentage}% Possibility of Glaucoma", risk_percentage
    
    else:  # CDR ≥ 0.5
        return "Glaucoma Detected", 100

# ✅ UI
@app.route('/')
def landing():
    return render_template("index.html")   # Landing page

@app.route('/detect')
def home():
    return render_template("detect.html")  # Glaucoma detection UI
    
# ✅ File Upload Handler
@app.route('/predict', methods=['POST'])
def predict():
    input_file = request.files.get("file")
    if not input_file:
        return jsonify({"error": "No file uploaded."}), 400

    file_ext = input_file.filename.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
        input_file.save(temp_file.name)
        file_path = temp_file.name

    if file_ext in ['jpg', 'jpeg', 'png']:
        frame = cv2.imread(file_path)
        cdr_value, conf = process_frame(frame)
    elif file_ext in ['mp4', 'avi']:
        cdr_value = calculate_cdr(file_path)
    else:
        return jsonify({"error": "Unsupported file type."}), 400

    diagnosis, risk = get_glaucoma_risk(cdr_value)

    if cdr_value is None:
        return jsonify({
            "cdr": None,
            "diagnosis": "No valid CDR detected",
            "risk": None
        })
    
    return jsonify({
        "cdr": round(cdr_value, 3),
        "diagnosis": diagnosis,
        "risk": risk
    })

# ✅ Start Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
