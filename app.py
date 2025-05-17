# app.py
import os
import json
import torch
import numpy as np
import urllib.request
from PIL import Image
from datetime import datetime
from torchvision import transforms
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
from io import BytesIO
from pymongo import MongoClient
import uuid

# Import the DamageAssessmentModel from your inference.py file
from inference import DamageAssessmentModel, load_model

app = Flask(__name__)
# Enable CORS for all routes and origins (important for mobile app connection)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MODEL_PATH'] = 'final_model.pt'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# MongoDB Atlas connection setup
# Replace with your MongoDB Atlas connection string
mongodb_uri = os.environ.get('MONGODB_URI', 'mongodb+srv://faahadjameel:Fahad@123@cluster1.48sdnhj.mongodb.net/?retryWrites=true&w=majority&appName=cluster1')
client = MongoClient(mongodb_uri)
db = client['disaster_assessment']
assessments_collection = db['assessments']

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
port = int(os.environ.get('PORT', 10000))
# Function to download model if not present
def download_model():
    model_path = app.config['MODEL_PATH']
    if not os.path.exists(model_path):
        print("Downloading model file...")
        try:
            model_url = 'https://drive.google.com/uc?export=download&id=18MKflSfZq4VblO6HhxlqpPFIDJhMUSSh'
            urllib.request.urlretrieve(model_url, model_path)
            print("Model downloaded successfully")
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            print("Using dummy model for testing")
            return None
    return model_path

# Download and load the model at startup
print(f"Loading model...")
try:
    model_path = download_model()
    model, config, damage_types = load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Still create a dummy model and config for testing if needed
    model = None
    config = {}
    damage_types = ["structural damage", "flooding", "water damage", "severe damage", 
                    "moderate damage", "mild damage", "roof damage", "wall damage"]
    print("Using dummy model for testing")

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def run_inference(model, image_path, damage_types, building_info=None, img_size=224):
    """Run inference on a single image with additional building information."""
    # Prepare image transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # If model is None (for testing), generate random outputs
    if model is None:
        print("Using dummy inference")
        severity = np.random.random() * 0.7  # Random severity between 0 and 0.7
        detected_damage_types = np.random.choice(damage_types, size=np.random.randint(0, 3), replace=False).tolist()
        classification_details = [0.3 + np.random.random() * 0.2 for _ in range(3)]
        classification_details = classification_details / np.sum(classification_details)
    else:
        # Run inference with the actual model
        with torch.no_grad():
            outputs = model(image_tensor)
        
        # Process outputs
        severity = outputs["severity"].item()
        damage_probs = torch.sigmoid(outputs["damage_types"]).numpy()[0]
        classification_details = outputs["classification_details"].numpy()[0]
        
        # Get damage types above threshold
        detected_damage_types = []
        for i, prob in enumerate(damage_probs):
            if prob > 0.5:  # Threshold for positive classification
                detected_damage_types.append(damage_types[i])
    
    # Create classification details dict
    details_dict = {
        "flood damage": float(classification_details[0]),
        "severe building damage": float(classification_details[1]),
        "moderate building damage": float(classification_details[2])
    }
    
    # Use provided building info if available
    building_name = building_info.get('name', 'B001') if building_info else 'B001'
    lat = building_info.get('latitude', '') if building_info else ''
    lng = building_info.get('longitude', '') if building_info else ''
    region = building_info.get('location', '') if building_info else ''
    
    # Create final output format
    result = {
        "Frame_Name": os.path.basename(image_path),
        "Buildings": [
            [building_name, lat, lng, severity, 1, 1, ", ".join(detected_damage_types)]
        ],
        "Capture date": datetime.now().strftime("%Y-%m-%d"),
        "Region": region,
        "Damage_Assessment": {
            "severity": float(severity),
            "damage_types": detected_damage_types,
            "assessment_date": datetime.now().strftime("%Y-%m-%d"),
            "model_used": "CNN Ensemble",
            "classification_details": details_dict
        }
    }
    
    return result

def highlight_damage(image_path, severity):
    """Apply a visual highlight to potentially damaged areas based on severity."""
    # Load the image
    img = cv2.imread(image_path)
    
    # Create a heat map overlay based on severity
    # This is a simplified version - in a real application you would use 
    # more sophisticated methods like Grad-CAM or a separate segmentation model
    heat_map = np.ones(img.shape, dtype=np.float32)
    
    # More severe damage gets more intense red highlight
    heat_map[:,:,0] = 1.0  # Blue channel
    heat_map[:,:,1] = 1.0 - severity * 0.7  # Green channel (reduce for more red)
    heat_map[:,:,2] = 1.0 - severity * 0.7  # Red channel (reduce for more blue)
    
    # Apply the heat map with alpha blending
    alpha = 0.3 + severity * 0.3  # Severity affects visibility
    highlighted_img = cv2.addWeighted(img, 1 - alpha, (img * heat_map).astype(np.uint8), alpha, 0)
    
    # Save the highlighted image to a BytesIO object
    is_success, buffer = cv2.imencode(".jpg", highlighted_img)
    io_buf = BytesIO(buffer)
    io_buf.seek(0)
    
    return io_buf

@app.route('/api/ping', methods=['GET'])
def ping():
    """Simple ping endpoint to test connectivity"""
    return jsonify({"status": "ok", "message": "API is running"})

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok", "message": "Service is running"})

@app.route('/api/assess', methods=['POST'])
def assess_damage():
    """API endpoint to assess building damage from uploaded image"""
    print("Received assessment request")
    
    # Debug incoming request
    print(f"Files in request: {request.files}")
    print(f"Form data: {request.form}")
    
    # Check if post-disaster image file is present in request
    if 'image' not in request.files:
        return jsonify({"error": "No post-disaster image provided"}), 400
    
    file = request.files['image']
    
    # Check if the file is valid
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Supported types: {', '.join(app.config['ALLOWED_EXTENSIONS'])}"}), 400
    
    # Get building information from form data
    building_info = {
        'name': request.form.get('building_name', 'B001'),
        'latitude': request.form.get('latitude', ''),
        'longitude': request.form.get('longitude', ''),
        'location': request.form.get('location', '')
    }
    
    print(f"Building info: {building_info}")
    
    # Save the post-disaster image
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    print(f"Saved post-disaster image to {filepath}")
    
    # Check if pre-disaster image is provided (optional)
    pre_disaster_filename = None
    if 'pre_disaster_image' in request.files and request.files['pre_disaster_image'].filename != '':
        pre_file = request.files['pre_disaster_image']
        if allowed_file(pre_file.filename):
            pre_disaster_filename = secure_filename(f"pre_{pre_file.filename}")
            pre_disaster_filepath = os.path.join(app.config['UPLOAD_FOLDER'], pre_disaster_filename)
            pre_file.save(pre_disaster_filepath)
            print(f"Saved pre-disaster image to {pre_disaster_filepath}")
    
    try:
        # Run damage assessment inference ONLY on post-disaster image
        result = run_inference(
            model, 
            filepath, 
            damage_types, 
            building_info=building_info,
            img_size=config.get('IMG_SIZE', 224) if config else 224
        )
        
        # Add image URLs to result
        # In production, we'd use absolute URLs with the host domain
        host_url = request.host_url.rstrip('/')
        result['image_url'] = f"/api/image/{filename}"
        result['highlighted_image_url'] = f"/api/highlighted/{filename}"
        
        # Add pre-disaster image URL if available
        if pre_disaster_filename:
            result['pre_disaster_image_url'] = f"/api/image/{pre_disaster_filename}"
        
        print(f"Assessment complete: {result}")
        return jsonify(result)
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

@app.route('/api/save-assessment', methods=['POST'])
def save_assessment():
    """Save the complete assessment including user input to MongoDB"""
    try:
        data = request.json
        
        # Generate a unique ID for this assessment
        assessment_id = str(uuid.uuid4())
        
        # Create assessment document
        assessment = {
            "_id": assessment_id,
            "building_info": {
                "name": data.get("building_name", ""),
                "location": data.get("location", ""),
                "latitude": data.get("latitude", ""),
                "longitude": data.get("longitude", "")
            },
            "model_assessment": data.get("model_assessment", {}),
            "user_assessment": {
                "severity": data.get("user_severity", 0),
                "comments": data.get("user_comments", "")
            },
            "image_urls": {
                "post_disaster": data.get("post_disaster_image", ""),
                "pre_disaster": data.get("pre_disaster_image", ""),
                "highlighted": data.get("highlighted_image", "")
            },
            "created_at": datetime.now()
        }
        
        # Insert into MongoDB
        result = assessments_collection.insert_one(assessment)
        
        return jsonify({
            "success": True, 
            "assessment_id": assessment_id,
            "message": "Assessment saved successfully to database"
        })
    
    except Exception as e:
        print(f"Error saving assessment: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error saving assessment: {str(e)}"}), 500

@app.route('/api/image/<filename>', methods=['GET'])
def get_image(filename):
    """Return the original uploaded image"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        return send_file(filepath, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": f"Error retrieving image: {str(e)}"}), 404

@app.route('/api/highlighted/<filename>', methods=['GET'])
def get_highlighted_image(filename):
    """Return the image with damage highlighted"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        
        # Get the severity from a previous assessment or run inference again
        result = run_inference(model, filepath, damage_types)
        severity = result['Damage_Assessment']['severity']
        
        # Generate highlighted image
        highlighted_img_io = highlight_damage(filepath, severity)
        
        return send_file(highlighted_img_io, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": f"Error highlighting image: {str(e)}"}), 500

@app.route('/api/assessments', methods=['GET'])
def get_assessments():
    """Retrieve all assessments from the database"""
    try:
        # Get limit parameter, default to 100
        limit = int(request.args.get('limit', 100))
        
        # Retrieve assessments from MongoDB
        cursor = assessments_collection.find().sort('created_at', -1).limit(limit)
        
        # Convert to list and format dates
        assessments = []
        for doc in cursor:
            # Convert ObjectId to string for JSON serialization
            doc['_id'] = str(doc['_id'])
            # Convert datetime to string
            if 'created_at' in doc:
                doc['created_at'] = doc['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            assessments.append(doc)
        
        return jsonify({
            "success": True,
            "count": len(assessments),
            "assessments": assessments
        })
    
    except Exception as e:
        print(f"Error retrieving assessments: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error retrieving assessments: {str(e)}"}), 500

@app.route('/api/assessment/<assessment_id>', methods=['GET'])
def get_assessment(assessment_id):
    """Retrieve a specific assessment by ID"""
    try:
        # Find the assessment in MongoDB
        assessment = assessments_collection.find_one({"_id": assessment_id})
        
        if not assessment:
            return jsonify({"error": "Assessment not found"}), 404
        
        # Convert ObjectId to string for JSON serialization
        assessment['_id'] = str(assessment['_id'])
        # Convert datetime to string
        if 'created_at' in assessment:
            assessment['created_at'] = assessment['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            "success": True,
            "assessment": assessment
        })
    
    except Exception as e:
        print(f"Error retrieving assessment: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error retrieving assessment: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the Flask server on all network interfaces 
    # so it's accessible from the mobile app
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
