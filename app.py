from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import keras
import matplotlib.pyplot as plt
from skimage import io
import threading

# Define and register the focal_tversky function
@keras.saving.register_keras_serializable()
def focal_tversky(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75):
    # Your implementation of focal_tversky function
    pass

# Initialize Flask app
app = Flask(__name__)

# Load the classifier model
classifier_model = load_model('C:/Users/desil/OneDrive/Desktop/WoodNEw/classifier-resnet-weights.keras')

# Load the segmentation model
segmentation_model = load_model('C:/Users/desil/OneDrive/Desktop/WoodNEw/ResUNet-weights.keras')

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'C:/Users/desil/OneDrive/Desktop/WoodNEw/Upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to visualize the image and mask
def visualize_image_and_mask(image, predicted_mask):
    # Convert image to suitable depth
    image = cv2.convertScaleAbs(image)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 2, 2)
    plt.title('Predicted Mask')
    plt.imshow(predicted_mask, cmap='gray')
    
    plt.axis('off')

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html', result=None)

# Route for handling file upload and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # Check if the file has a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Check if the file is allowed
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save the file to the upload folder
        file.save(filepath)
        
        # Load the image and preprocess it
        image = cv2.imread(filepath)
        image = cv2.resize(image, (256, 256))  # Resize image to match model input size
        image = image / 255.0  # Normalize pixel values
        
        # Make prediction with the classifier model
        classifier_prediction = classifier_model.predict(np.array([image]))
        defect_probability = classifier_prediction[0][1]  # Probability of defect class
        
        if defect_probability < 0.5:
            return render_template('index.html', result=f'defect detected (Confidence: {defect_probability:.2f})')
        else:
            # Make prediction with the segmentation model
            segmentation_prediction = segmentation_model.predict(np.array([image]))
            predicted_mask = segmentation_prediction[0].squeeze().round()
            
            # Visualize the original image and predicted mask
            visualize_image_and_mask(image, predicted_mask)
            
            # Save the result image path
            result_image_path = os.path.join('C:/Users/desil/OneDrive/Desktop/WoodNEw/Upload', filename)
            
            # Return both the result and the confidence score
            return render_template('index.html', result=f'No Defect detected (Confidence: {defect_probability:.2f})', result_image=result_image_path)
    
    return jsonify({'error': 'File not allowed or unsupported format'})

if __name__ == '__main__':
    app.run(debug=True)
