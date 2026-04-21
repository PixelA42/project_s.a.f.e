from flask import Flask, request, jsonify
import os
import uuid
import numpy as np
import librosa
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# ==========================================
# CONFIGURATION & SECURITY
# ==========================================
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB Limit
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a'}

# Create a temporary folder for processing uploads
TEMP_DIR = 'temp_processing'
os.makedirs(TEMP_DIR, exist_ok=True)

# Load the actual trained AI model! (Loads once when server starts)
print("🧠 Loading S.A.F.E. Neural Network...")
try:
    MODEL = tf.keras.models.load_model('safe_model.keras')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ WARNING: Could not load safe_model.keras. Did you train it yet? Error: {e}")
    MODEL = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==========================================
# AUDIO PROCESSING PIPELINE (For API)
# ==========================================
def process_audio_for_prediction(audio_path, image_path):
    """Converts a single audio file to a spectrogram for the CNN."""
    # 1. Load Audio
    y, sr = librosa.load(audio_path, sr=22050, duration=5.0)
    
    # 2. Convert to Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # 3. Save as Image (224x224 pixels)
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    librosa.display.specshow(mel_db, sr=sr, fmax=8000, cmap='magma')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 4. Format Image for the Neural Network
    img = load_img(image_path, target_size=(224, 224), color_mode='rgb')
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    
    return img_array

# ==========================================
# MAIN ROUTE
# ==========================================
@app.route('/detect-audio', methods=['POST'])
def detect_audio():
    # 1. Validation Edge Cases
    if 'file' not in request.files:
        return jsonify({'error': 'Bad Request: No file attached'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Bad Request: No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported Media Type: Only .mp3, .wav, .flac, or .m4a allowed'}), 415

    if MODEL is None:
        return jsonify({'error': 'Internal Server Error: AI Model is offline.'}), 500

    # 2. Secure File Saving
    # Use UUID to prevent two users uploading "test.mp3" at the exact same time and overriding each other
    unique_id = str(uuid.uuid4())
    safe_filename = secure_filename(file.filename)
    audio_path = os.path.join(TEMP_DIR, f"{unique_id}_{safe_filename}")
    image_path = os.path.join(TEMP_DIR, f"{unique_id}.png")
    
    try:
        # Save the uploaded audio to the temp folder
        file.save(audio_path)

        # 3. Process & Predict
        model_input = process_audio_for_prediction(audio_path, image_path)
        prediction_value = MODEL.predict(model_input)[0][0] # Get the raw probability math
        
        # 4. Format the Answer
        # Remember our labels: 0 = Human, 1 = AI
        is_fake = prediction_value > 0.5
        confidence = prediction_value if is_fake else (1.0 - prediction_value)
        
        result = {
            'status': 'success',
            'filename': safe_filename,
            'prediction': 'AI_Generated' if is_fake else 'Human_Voice',
            'confidence_score': round(float(confidence) * 100, 2), # Returns a clean percentage like 98.45%
            'risk_level': 'CRITICAL' if is_fake else 'SAFE'
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': f'Internal Server Error: Processing failed. Details: {str(e)}'}), 500

    finally:
        # 5. The Janitor: Always clean up files, even if the processing crashes!
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(image_path):
            os.remove(image_path)

# ==========================================
# HEALTH CHECK
# ==========================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'online', 'project': 'S.A.F.E.', 'model_status': 'Loaded' if MODEL else 'Offline'}), 200

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'Payload Too Large: File exceeds the 5MB limit.'}), 413

if __name__ == '__main__':
    print("🚀 S.A.F.E. Production API is online...")
    app.run(debug=True, port=5000)