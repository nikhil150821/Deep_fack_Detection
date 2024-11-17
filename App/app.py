from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from PIL import Image  # For image processing
import cv2  # For video processing
import torch
from model import tModel, prepare_image, prepare_video, predict_image, predict_video, load_model

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flash messages

# Define allowed extensions and temporary directory
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'jpg', 'jpeg', 'png'}
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Utility function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file was uploaded
        if 'media' not in request.files:
            flash("No file part in the request", "error")
            return redirect(request.url)
        
        media_file = request.files['media']

        # Check if the file is selected and has an allowed extension
        if media_file.filename == '' or not allowed_file(media_file.filename):
            flash("No selected file or unsupported file type. Please upload an image or video.", "error")
            return redirect(request.url)
        
        filename = secure_filename(media_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        media_file.save(file_path)
        
        # Process based on file type
        file_extension = filename.rsplit('.', 1)[1].lower()
        
        if file_extension in {'jpg', 'jpeg', 'png'}:
            # Process image file
            output = predict_image(model, file_path)
            return render_template("predict.html", file_type="image", uploaded_image=filename, output=output)
        
        elif file_extension in {'mp4', 'mov', 'avi'}:
            # Process video file
            faces_detected, output = predict_video(model, file_path)  # Modify predict_video to return face detection status
            if not faces_detected:
                flash("No faces detected in the video. Cannot process.", "error")
                return redirect(request.url)
            return render_template("predict.html", file_type="video", uploaded_video=filename, output=output)
        
        else:
            flash("Unsupported file type", "error")
            return redirect(request.url)

    return render_template("index.html")

if __name__ == "__main__":
    # Load model
    model_path = 'model_97_acc_80_frames_FF_data.pt'
    model = load_model(model_path, num_classes=2)  # Adjust num_classes as needed

    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    app.run(debug=True)

# don't change any thing new just add code such that face detection function works on the image or video that is uploaded on this app.py
# and plot a box on the face detected for both the image and video aswell
# condition :if face detected only it is ready to get prediction else print  no face detected
